//===-- SBReproducer.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SBReproducerPrivate.h"

#include "lldb/API/LLDB.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBAttachInfo.h"
#include "lldb/API/SBBlock.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBData.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBReproducer.h"

#include "lldb/Host/FileSystem.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::repro;

SBRegistry::SBRegistry() {}

const char *SBReproducer::Capture(const char *path) {
  static std::string error;
  if (auto e =
          Reproducer::Initialize(ReproducerMode::Capture, FileSpec(path))) {
    error = llvm::toString(std::move(e));
    return error.c_str();
  }
  return nullptr;
}

const char *SBReproducer::Replay(const char *path) {
  static std::string error;
  if (auto e = Reproducer::Initialize(ReproducerMode::Replay, FileSpec(path))) {
    error = llvm::toString(std::move(e));
    return error.c_str();
  }

  repro::Loader *loader = repro::Reproducer::Instance().GetLoader();
  if (!loader) {
    error = "unable to get replay loader.";
    return error.c_str();
  }

  // FIXME: Enable the following code once the SB reproducer has landed.
#if 0
  FileSpec file = loader->GetFile<SBInfo>();
  if (!file) {
    error = "unable to get replay data from reproducer.";
    return error.c_str();
  }

  SBRegistry registry;
  registry.Replay(file);
#endif

  return nullptr;
}

char lldb_private::repro::SBProvider::ID = 0;
const char *SBInfo::name = "sbapi";
const char *SBInfo::file = "sbapi.bin";
