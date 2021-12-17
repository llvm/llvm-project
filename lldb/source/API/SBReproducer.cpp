//===-- SBReproducer.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SBReproducerPrivate.h"

#include "lldb/API/LLDB.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBAttachInfo.h"
#include "lldb/API/SBBlock.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandInterpreterRunOptions.h"
#include "lldb/API/SBData.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBReproducer.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Version/Version.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::repro;

SBReplayOptions::SBReplayOptions()
    : m_opaque_up(std::make_unique<ReplayOptions>()){}

SBReplayOptions::SBReplayOptions(const SBReplayOptions &rhs)
    : m_opaque_up(std::make_unique<ReplayOptions>(*rhs.m_opaque_up)) {}

SBReplayOptions::~SBReplayOptions() = default;

SBReplayOptions &SBReplayOptions::operator=(const SBReplayOptions &rhs) {
  if (this == &rhs)
    return *this;
  *m_opaque_up = *rhs.m_opaque_up;
  return *this;
}

void SBReplayOptions::SetVerify(bool verify) { m_opaque_up->verify = verify; }

bool SBReplayOptions::GetVerify() const { return m_opaque_up->verify; }

void SBReplayOptions::SetCheckVersion(bool check) {
  m_opaque_up->check_version = check;
}

bool SBReplayOptions::GetCheckVersion() const {
  return m_opaque_up->check_version;
}

SBRegistry::SBRegistry() {
  Registry &R = *this;

  RegisterMethods<SBAddress>(R);
  RegisterMethods<SBAttachInfo>(R);
  RegisterMethods<SBBlock>(R);
  RegisterMethods<SBBreakpoint>(R);
  RegisterMethods<SBBreakpointList>(R);
  RegisterMethods<SBBreakpointLocation>(R);
  RegisterMethods<SBBreakpointName>(R);
  RegisterMethods<SBBroadcaster>(R);
  RegisterMethods<SBCommandInterpreter>(R);
  RegisterMethods<SBCommandInterpreterRunOptions>(R);
  RegisterMethods<SBCommandReturnObject>(R);
  RegisterMethods<SBCommunication>(R);
  RegisterMethods<SBCompileUnit>(R);
  RegisterMethods<SBData>(R);
  RegisterMethods<SBDebugger>(R);
  RegisterMethods<SBDeclaration>(R);
  RegisterMethods<SBEnvironment>(R);
  RegisterMethods<SBError>(R);
  RegisterMethods<SBEvent>(R);
  RegisterMethods<SBExecutionContext>(R);
  RegisterMethods<SBExpressionOptions>(R);
  RegisterMethods<SBFile>(R);
  RegisterMethods<SBFileSpec>(R);
  RegisterMethods<SBFileSpecList>(R);
  RegisterMethods<SBFrame>(R);
  RegisterMethods<SBFunction>(R);
  RegisterMethods<SBHostOS>(R);
  RegisterMethods<SBInputReader>(R);
  RegisterMethods<SBInstruction>(R);
  RegisterMethods<SBInstructionList>(R);
  RegisterMethods<SBLanguageRuntime>(R);
  RegisterMethods<SBLaunchInfo>(R);
  RegisterMethods<SBLineEntry>(R);
  RegisterMethods<SBListener>(R);
  RegisterMethods<SBMemoryRegionInfo>(R);
  RegisterMethods<SBMemoryRegionInfoList>(R);
  RegisterMethods<SBModule>(R);
  RegisterMethods<SBModuleSpec>(R);
  RegisterMethods<SBPlatform>(R);
  RegisterMethods<SBPlatformConnectOptions>(R);
  RegisterMethods<SBPlatformShellCommand>(R);
  RegisterMethods<SBProcess>(R);
  RegisterMethods<SBProcessInfo>(R);
  RegisterMethods<SBQueue>(R);
  RegisterMethods<SBQueueItem>(R);
  RegisterMethods<SBSection>(R);
  RegisterMethods<SBSourceManager>(R);
  RegisterMethods<SBStream>(R);
  RegisterMethods<SBStringList>(R);
  RegisterMethods<SBStructuredData>(R);
  RegisterMethods<SBSymbol>(R);
  RegisterMethods<SBSymbolContext>(R);
  RegisterMethods<SBSymbolContextList>(R);
  RegisterMethods<SBTarget>(R);
  RegisterMethods<SBThread>(R);
  RegisterMethods<SBThreadCollection>(R);
  RegisterMethods<SBThreadPlan>(R);
  RegisterMethods<SBTrace>(R);
  RegisterMethods<SBType>(R);
  RegisterMethods<SBTypeCategory>(R);
  RegisterMethods<SBTypeEnumMember>(R);
  RegisterMethods<SBTypeFilter>(R);
  RegisterMethods<SBTypeFormat>(R);
  RegisterMethods<SBTypeNameSpecifier>(R);
  RegisterMethods<SBTypeSummary>(R);
  RegisterMethods<SBTypeSummaryOptions>(R);
  RegisterMethods<SBTypeSynthetic>(R);
  RegisterMethods<SBUnixSignals>(R);
  RegisterMethods<SBValue>(R);
  RegisterMethods<SBValueList>(R);
  RegisterMethods<SBVariablesOptions>(R);
  RegisterMethods<SBWatchpoint>(R);
}

const char *SBReproducer::Capture() {
  static std::string error;
  if (auto e = Reproducer::Initialize(ReproducerMode::Capture, llvm::None)) {
    error = llvm::toString(std::move(e));
    return error.c_str();
  }

  if (auto *g = lldb_private::repro::Reproducer::Instance().GetGenerator()) {
    auto &p = g->GetOrCreate<SBProvider>();
    InstrumentationData::Initialize(p.GetSerializer(), p.GetRegistry());
  }

  return nullptr;
}

const char *SBReproducer::Capture(const char *path) {
  static std::string error;
  if (auto e =
          Reproducer::Initialize(ReproducerMode::Capture, FileSpec(path))) {
    error = llvm::toString(std::move(e));
    return error.c_str();
  }

  if (auto *g = lldb_private::repro::Reproducer::Instance().GetGenerator()) {
    auto &p = g->GetOrCreate<SBProvider>();
    InstrumentationData::Initialize(p.GetSerializer(), p.GetRegistry());
  }

  return nullptr;
}

const char *SBReproducer::PassiveReplay(const char *path) {
  return "Reproducer replay has been removed";
}

const char *SBReproducer::Replay(const char *path) {
  return "Reproducer replay has been removed";
}

const char *SBReproducer::Replay(const char *path, bool skip_version_check) {
  return Replay(path);
}

const char *SBReproducer::Replay(const char *path,
                                 const SBReplayOptions &options) {
  return Replay(path);
}

const char *SBReproducer::Finalize(const char *path) {
  static std::string error;

  repro::Loader *loader = repro::Reproducer::Instance().GetLoader();
  if (!loader) {
    error = "unable to get replay loader.";
    return error.c_str();
  }

  if (auto e = repro::Finalize(loader)) {
    error = llvm::toString(std::move(e));
    return error.c_str();
  }

  return nullptr;
}

bool SBReproducer::Generate() {
  auto &r = Reproducer::Instance();
  if (auto generator = r.GetGenerator()) {
    generator->Keep();
    return true;
  }
  return false;
}

bool SBReproducer::SetAutoGenerate(bool b) {
  auto &r = Reproducer::Instance();
  if (auto generator = r.GetGenerator()) {
    generator->SetAutoGenerate(b);
    return true;
  }
  return false;
}

const char *SBReproducer::GetPath() {
  ConstString path;
  auto &r = Reproducer::Instance();
  if (FileSpec reproducer_path = Reproducer::Instance().GetReproducerPath())
    path = ConstString(r.GetReproducerPath().GetCString());
  return path.GetCString();
}

void SBReproducer::SetWorkingDirectory(const char *path) {
  if (auto *g = lldb_private::repro::Reproducer::Instance().GetGenerator()) {
    auto &wp = g->GetOrCreate<repro::WorkingDirectoryProvider>();
    wp.SetDirectory(path);
    auto &fp = g->GetOrCreate<repro::FileProvider>();
    fp.RecordInterestingDirectory(wp.GetDirectory());
  }
}

char lldb_private::repro::SBProvider::ID = 0;
const char *SBProvider::Info::name = "sbapi";
const char *SBProvider::Info::file = "sbapi.bin";
