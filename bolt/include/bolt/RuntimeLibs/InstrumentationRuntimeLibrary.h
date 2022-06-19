//===- bolt/RuntimeLibs/InstrumentationRuntimeLibrary.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the InstrumentationRuntimeLibrary
// class.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_RUNTIMELIBS_INSTRUMENTATION_RUNTIME_LIBRARY_H
#define BOLT_RUNTIMELIBS_INSTRUMENTATION_RUNTIME_LIBRARY_H

#include "bolt/Passes/InstrumentationSummary.h"
#include "bolt/RuntimeLibs/RuntimeLibrary.h"
#include <memory>

namespace llvm {
namespace bolt {

class InstrumentationRuntimeLibrary : public RuntimeLibrary {
public:
  void setSummary(std::unique_ptr<InstrumentationSummary> &&S) {
    Summary.swap(S);
  }

  void addRuntimeLibSections(std::vector<std::string> &SecNames) const final {
    SecNames.push_back(".bolt.instr.counters");
  }

  void adjustCommandLineOptions(const BinaryContext &BC) const final;

  void emitBinary(BinaryContext &BC, MCStreamer &Streamer) final;

  void link(BinaryContext &BC, StringRef ToolPath, RuntimeDyld &RTDyld,
            std::function<void(RuntimeDyld &)> OnLoad) final;

private:
  std::string buildTables(BinaryContext &BC);

  /// Create a non-allocatable ELF section with read-only tables necessary for
  /// writing the instrumented data profile during program finish. The runtime
  /// library needs to open the program executable file and read this data from
  /// disk, this is not loaded by the system.
  void emitTablesAsELFNote(BinaryContext &BC);

  std::unique_ptr<InstrumentationSummary> Summary;
};

} // namespace bolt
} // namespace llvm

#endif
