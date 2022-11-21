//===- bolt/RuntimeLibs/HugifyRuntimeLibrary.cpp - Hugify RT Library ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HugifyRuntimeLibrary class.
//
//===----------------------------------------------------------------------===//

#include "bolt/RuntimeLibs/HugifyRuntimeLibrary.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> HotText;

cl::opt<bool>
    Hugify("hugify",
           cl::desc("Automatically put hot code on 2MB page(s) (hugify) at "
                    "runtime. No manual call to hugify is needed in the binary "
                    "(which is what --hot-text relies on)."),
           cl::cat(BoltOptCategory));

static cl::opt<std::string> RuntimeHugifyLib(
    "runtime-hugify-lib",
    cl::desc("specify file name of the runtime hugify library"),
    cl::init("libbolt_rt_hugify.a"), cl::cat(BoltOptCategory));

} // namespace opts

void HugifyRuntimeLibrary::adjustCommandLineOptions(
    const BinaryContext &BC) const {
  if (opts::HotText) {
    errs()
        << "BOLT-ERROR: -hot-text should be applied to binaries with "
           "pre-compiled manual hugify support, while -hugify will add hugify "
           "support automatically. These two options cannot both be present.\n";
    exit(1);
  }
  // After the check, we set HotText to be true because automated hugify support
  // relies on it.
  opts::HotText = true;
  if (!BC.StartFunctionAddress) {
    errs() << "BOLT-ERROR: hugify runtime libraries require a known entry "
              "point of "
              "the input binary\n";
    exit(1);
  }
}

void HugifyRuntimeLibrary::link(BinaryContext &BC, StringRef ToolPath,
                                RuntimeDyld &RTDyld,
                                std::function<void(RuntimeDyld &)> OnLoad) {
  std::string LibPath = getLibPath(ToolPath, opts::RuntimeHugifyLib);
  loadLibrary(LibPath, RTDyld);
  OnLoad(RTDyld);
  RTDyld.finalizeWithMemoryManagerLocking();
  if (RTDyld.hasError()) {
    outs() << "BOLT-ERROR: RTDyld failed: " << RTDyld.getErrorString() << "\n";
    exit(1);
  }

  assert(!RuntimeStartAddress &&
         "We don't currently support linking multiple runtime libraries");
  RuntimeStartAddress = RTDyld.getSymbol("__bolt_hugify_self").getAddress();
  if (!RuntimeStartAddress) {
    errs() << "BOLT-ERROR: instrumentation library does not define "
              "__bolt_hugify_self: "
           << LibPath << "\n";
    exit(1);
  }
}
