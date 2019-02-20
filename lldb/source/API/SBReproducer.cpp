//===-- SBReproducer.cpp ----------------------------------------*- C++ -*-===//
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

bool SBReproducer::Replay() {
  repro::Loader *loader = repro::Reproducer::Instance().GetLoader();
  if (!loader)
    return false;

  FileSpec file = loader->GetFile<SBInfo>();
  if (!file)
    return false;

  SBRegistry registry;
  registry.Replay(file);

  return true;
}

char lldb_private::repro::SBProvider::ID = 0;
const char *SBInfo::name = "sbapi";
const char *SBInfo::file = "sbapi.bin";
