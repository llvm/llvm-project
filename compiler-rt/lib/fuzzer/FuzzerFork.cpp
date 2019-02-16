//===- FuzzerFork.cpp - run fuzzing in separate subprocesses --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Spawn and orchestrate separate fuzzing processes.
//===----------------------------------------------------------------------===//

#include "FuzzerCommand.h"
#include "FuzzerFork.h"
#include "FuzzerIO.h"
#include "FuzzerMerge.h"
#include "FuzzerSHA1.h"
#include "FuzzerTracePC.h"
#include "FuzzerUtil.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

namespace fuzzer {

struct Stats {
  size_t number_of_executed_units = 0;
  size_t peak_rss_mb = 0;
  size_t average_exec_per_sec = 0;
};

static Stats ParseFinalStatsFromLog(const std::string &LogPath) {
  std::ifstream In(LogPath);
  std::string Line;
  Stats Res;
  struct {
    const char *Name;
    size_t *Var;
  } NameVarPairs[] = {
      {"stat::number_of_executed_units:", &Res.number_of_executed_units},
      {"stat::peak_rss_mb:", &Res.peak_rss_mb},
      {"stat::average_exec_per_sec:", &Res.average_exec_per_sec},
      {nullptr, nullptr},
  };
  while (std::getline(In, Line, '\n')) {
    if (Line.find("stat::") != 0) continue;
    std::istringstream ISS(Line);
    std::string Name;
    size_t Val;
    ISS >> Name >> Val;
    for (size_t i = 0; NameVarPairs[i].Name; i++)
      if (Name == NameVarPairs[i].Name)
        *NameVarPairs[i].Var = Val;
  }
  return Res;
}

struct FuzzJob {
  // Inputs.
  Command Cmd;
  std::string CorpusDir;
  std::string LogPath;
  std::string CFPath;

  // Fuzzing Outputs.
  int ExitCode;
};

struct GlobalEnv {
  Vector<std::string> Args;
  Vector<std::string> CorpusDirs;
  std::string MainCorpusDir;
  std::string TempDir;
  Set<uint32_t> Features, Cov;
  Vector<std::string> Files;
  Random *Rand;
  std::chrono::system_clock::time_point ProcessStartTime;
  int Verbosity = 0;

  size_t NumTimeouts = 0;
  size_t NumOOMs = 0;
  size_t NumCrashes = 0;


  size_t NumRuns = 0;

  size_t secondsSinceProcessStartUp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now() - ProcessStartTime)
        .count();
  }

  FuzzJob *CreateNewJob(size_t JobId) {
    Command Cmd(Args);
    Cmd.removeFlag("fork");
    for (auto &C : CorpusDirs) // Remove all corpora from the args.
      Cmd.removeArgument(C);
    Cmd.addFlag("reload", "0");  // working in an isolated dir, no reload.
    Cmd.addFlag("print_final_stats", "1");
    Cmd.addFlag("print_funcs", "0");  // no need to spend time symbolizing.
    Cmd.addFlag("max_total_time", std::to_string(std::min((size_t)300, JobId)));

    auto Job = new FuzzJob;
    std::string Seeds;
    if (size_t CorpusSubsetSize =
            std::min(Files.size(), (size_t)sqrt(Files.size() + 2)))
      for (size_t i = 0; i < CorpusSubsetSize; i++)
        Seeds += (Seeds.empty() ? "" : ",") +
                 Files[Rand->SkewTowardsLast(Files.size())];
    if (!Seeds.empty())
      Cmd.addFlag("seed_inputs", Seeds);
    Job->LogPath = DirPlusFile(TempDir, std::to_string(JobId) + ".log");
    Job->CorpusDir = DirPlusFile(TempDir, "C" + std::to_string(JobId));
    Job->CFPath = DirPlusFile(TempDir, std::to_string(JobId) + ".merge");


    Cmd.addArgument(Job->CorpusDir);
    RmDirRecursive(Job->CorpusDir);
    MkDir(Job->CorpusDir);

    Cmd.setOutputFile(Job->LogPath);
    Cmd.combineOutAndErr();

    Job->Cmd = Cmd;

    if (Verbosity >= 2)
      Printf("Job %zd/%p Created: %s\n", JobId, Job,
             Job->Cmd.toString().c_str());
    // Start from very short runs and gradually increase them.
    return Job;
  }

  void RunOneMergeJob(FuzzJob *Job) {
    Vector<SizedFile> TempFiles;
    GetSizedFilesFromDir(Job->CorpusDir, &TempFiles);

    Vector<std::string> FilesToAdd;
    Set<uint32_t> NewFeatures, NewCov;
    CrashResistantMerge(Args, {}, TempFiles, &FilesToAdd, Features,
                        &NewFeatures, Cov, &NewCov, Job->CFPath, false);
    RemoveFile(Job->CFPath);
    for (auto &Path : FilesToAdd) {
      auto U = FileToVector(Path);
      auto NewPath = DirPlusFile(MainCorpusDir, Hash(U));
      WriteToFile(U, NewPath);
      Files.push_back(NewPath);
    }
    RmDirRecursive(Job->CorpusDir);
    Features.insert(NewFeatures.begin(), NewFeatures.end());
    Cov.insert(NewCov.begin(), NewCov.end());
    for (auto Idx : NewCov)
      if (auto *TE = TPC.PCTableEntryByIdx(Idx))
        if (TPC.PcIsFuncEntry(TE))
          PrintPC("  NEW_FUNC: %p %F %L\n", "",
                  TPC.GetNextInstructionPc(TE->PC));

    auto Stats = ParseFinalStatsFromLog(Job->LogPath);
    NumRuns += Stats.number_of_executed_units;
    if (!FilesToAdd.empty() || Job->ExitCode != 0)
      Printf("#%zd: cov: %zd ft: %zd corp: %zd exec/s %zd "
             "oom/timeout/crash: %zd/%zd/%zd time: %zds\n", NumRuns,
             Cov.size(), Features.size(), Files.size(),
             Stats.average_exec_per_sec,
             NumOOMs, NumTimeouts, NumCrashes, secondsSinceProcessStartUp());
  }
};

struct JobQueue {
  std::queue<FuzzJob *> Qu;
  std::mutex Mu;

  void Push(FuzzJob *Job) {
    std::lock_guard<std::mutex> Lock(Mu);
    Qu.push(Job);
  }
  FuzzJob *Pop() {
    std::lock_guard<std::mutex> Lock(Mu);
    if (Qu.empty()) return nullptr;
    auto Job = Qu.front();
    Qu.pop();
    return Job;
  }
};

void WorkerThread(std::atomic<bool> *Stop, JobQueue *FuzzQ, JobQueue *MergeQ) {
  while (!Stop->load()) {
    auto Job = FuzzQ->Pop();
    // Printf("WorkerThread: job %p\n", Job);
    if (!Job) {
      SleepSeconds(1);
      continue;
    }
    Job->ExitCode = ExecuteCommand(Job->Cmd);
    MergeQ->Push(Job);
  }
}

// This is just a skeleton of an experimental -fork=1 feature.
void FuzzWithFork(Random &Rand, const FuzzingOptions &Options,
                  const Vector<std::string> &Args,
                  const Vector<std::string> &CorpusDirs, int NumJobs) {
  Printf("INFO: -fork=%d: fuzzing in separate process(s)\n", NumJobs);

  GlobalEnv Env;
  Env.Args = Args;
  Env.CorpusDirs = CorpusDirs;
  Env.Rand = &Rand;
  Env.Verbosity = Options.Verbosity;
  Env.ProcessStartTime = std::chrono::system_clock::now();

  Vector<SizedFile> SeedFiles;
  for (auto &Dir : CorpusDirs)
    GetSizedFilesFromDir(Dir, &SeedFiles);
  std::sort(SeedFiles.begin(), SeedFiles.end());
  Env.TempDir = TempPath(".dir");
  RmDirRecursive(Env.TempDir);  // in case there is a leftover from old runs.
  MkDir(Env.TempDir);


  if (CorpusDirs.empty())
    MkDir(Env.MainCorpusDir = DirPlusFile(Env.TempDir, "C"));
  else
    Env.MainCorpusDir = CorpusDirs[0];

  auto CFPath = DirPlusFile(Env.TempDir, "merge.txt");
  CrashResistantMerge(Env.Args, {}, SeedFiles, &Env.Files, {}, &Env.Features,
                      {}, &Env.Cov,
                      CFPath, false);
  RemoveFile(CFPath);
  Printf("INFO: -fork=%d: %zd seed inputs, starting to fuzz in %s\n", NumJobs,
         Env.Files.size(), Env.TempDir.c_str());

  int ExitCode = 0;

  JobQueue FuzzQ, MergeQ;
  std::atomic<bool> Stop(false);

  size_t JobId = 1;
  Vector<std::thread> Threads;
  for (int t = 0; t < NumJobs; t++) {
    Threads.push_back(std::thread(WorkerThread, &Stop, &FuzzQ, &MergeQ));
    FuzzQ.Push(Env.CreateNewJob(JobId++));
  }

  while (true) {
    auto Job = MergeQ.Pop();
    if (!Job) {
      if (Stop)
        break;
      SleepSeconds(1);
      continue;
    }
    ExitCode = Job->ExitCode;
    if (ExitCode != Options.InterruptExitCode)
      Env.RunOneMergeJob(Job);

    // Continue if our crash is one of the ignorred ones.
    if (Options.IgnoreTimeouts && ExitCode == Options.TimeoutExitCode)
      Env.NumTimeouts++;
    else if (Options.IgnoreOOMs && ExitCode == Options.OOMExitCode)
      Env.NumOOMs++;
    else if (ExitCode == Options.InterruptExitCode)
      Stop = true;
    else if (ExitCode != 0) {
      Env.NumCrashes++;
      if (Options.IgnoreCrashes) {
        std::ifstream In(Job->LogPath);
        std::string Line;
        while (std::getline(In, Line, '\n'))
          if (Line.find("ERROR:") != Line.npos)
            Printf("%s\n", Line.c_str());
      } else {
        // And exit if we don't ignore this crash.
        Printf("INFO: log from the inner process:\n%s",
               FileToString(Job->LogPath).c_str());
        Stop = true;
      }
    }
    RemoveFile(Job->LogPath);
    delete Job;

    // Stop if we are over the time budget.
    // This is not precise, since other threads are still running
    // and we will wait while joining them.
    // We also don't stop instantly: other jobs need to finish.
    if (Options.MaxTotalTimeSec > 0 && !Stop &&
        Env.secondsSinceProcessStartUp() >= (size_t)Options.MaxTotalTimeSec) {
      Printf("INFO: fuzzed for %zd seconds, wrapping up soon\n",
             Env.secondsSinceProcessStartUp());
      Stop = true;
    }

    if (!Stop)
      FuzzQ.Push(Env.CreateNewJob(JobId++));
  }
  Stop = true;

  for (auto &T : Threads)
    T.join();

  RmDirRecursive(Env.TempDir);

  // Use the exit code from the last child process.
  Printf("INFO: exiting: %d time: %zds\n", ExitCode,
         Env.secondsSinceProcessStartUp());
  exit(ExitCode);
}

} // namespace fuzzer

