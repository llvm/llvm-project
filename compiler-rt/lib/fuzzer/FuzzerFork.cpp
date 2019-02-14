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
#include "FuzzerUtil.h"

#include <atomic>
#include <mutex>
#include <thread>
#include <queue>

namespace fuzzer {

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
  Set<uint32_t> Features;
  Vector<std::string> Files;
  Random *Rand;
  int Verbosity = 0;

  FuzzJob *CreateNewJob(size_t JobId) {
    Command Cmd(Args);
    Cmd.removeFlag("fork");
    for (auto &C : CorpusDirs) // Remove all corpora from the args.
      Cmd.removeArgument(C);
    Cmd.addFlag("reload", "0");  // working in an isolated dir, no reload.
    Cmd.addFlag("max_total_time", std::to_string(std::min((size_t)300, JobId)));

    auto Job = new FuzzJob;
    std::string Seeds;
    if (size_t CorpusSubsetSize = std::min(Files.size(), (size_t)100))
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
    Set<uint32_t> NewFeatures;
    CrashResistantMerge(Args, {}, TempFiles, &FilesToAdd, Features,
                        &NewFeatures, Job->CFPath, false);
    RemoveFile(Job->CFPath);
    for (auto &Path : FilesToAdd) {
      auto U = FileToVector(Path);
      auto NewPath = DirPlusFile(MainCorpusDir, Hash(U));
      WriteToFile(U, NewPath);
      Files.push_back(NewPath);
    }
    Printf("Removing %s\n", Job->CorpusDir.c_str());
    RmDirRecursive(Job->CorpusDir);
    Features.insert(NewFeatures.begin(), NewFeatures.end());
    Printf("INFO: temp_files: %zd files_added: %zd newft: %zd ft: %zd\n",
           TempFiles.size(), FilesToAdd.size(), NewFeatures.size(),
           Features.size());
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
  Printf("INFO: -fork=%d: doing fuzzing in a separate process in order to "
         "be more resistant to crashes, timeouts, and OOMs\n", NumJobs);

  GlobalEnv Env;
  Env.Args = Args;
  Env.CorpusDirs = CorpusDirs;
  Env.Rand = &Rand;
  Env.Verbosity = Options.Verbosity;

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
                      CFPath, false);
  RemoveFile(CFPath);
  Printf("INFO: -fork=%d: %zd seeds, starting to fuzz; scratch: %s\n",
         NumJobs, Env.Files.size(), Env.TempDir.c_str());

  int ExitCode = 0;

  JobQueue FuzzQ, MergeQ;
  std::atomic<bool> Stop(false);

  size_t JobId = 1;
  Vector<std::thread> Threads;
  for (int t = 0; t < NumJobs; t++) {
    Threads.push_back(std::thread(WorkerThread, &Stop, &FuzzQ, &MergeQ));
    FuzzQ.Push(Env.CreateNewJob(JobId++));
  }

  while (!Stop) {
    auto Job = MergeQ.Pop();
    if (!Job) {
      SleepSeconds(1);
      continue;
    }
    ExitCode = Job->ExitCode;
    if (ExitCode != Options.InterruptExitCode)
      Env.RunOneMergeJob(Job);

    // Continue if our crash is one of the ignorred ones.
    if (Options.IgnoreTimeouts && ExitCode == Options.TimeoutExitCode)
      ;
    else if (Options.IgnoreOOMs && ExitCode == Options.OOMExitCode)
      ;
    else if (ExitCode == Options.InterruptExitCode)
      Stop = true;
    else if (ExitCode != 0) {
      // And exit if we don't ignore this crash.
      Printf("INFO: log from the inner process:\n%s",
                   FileToString(Job->LogPath).c_str());
      Stop = true;
    }
    RemoveFile(Job->LogPath);
    delete Job;
    FuzzQ.Push(Env.CreateNewJob(JobId++));
  }
  Stop = true;

  for (auto &T : Threads)
    T.join();

  RmDirRecursive(Env.TempDir);

  // Use the exit code from the last child process.
  Printf("Fork: exiting: %d\n", ExitCode);
  exit(ExitCode);
}

} // namespace fuzzer

