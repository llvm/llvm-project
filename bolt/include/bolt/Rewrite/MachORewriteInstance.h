//===- bolt/Rewrite/MachORewriteInstance.h - MachO rewriter -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to control an instance of a macho binary rewriting process.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_MACHO_REWRITE_INSTANCE_H
#define BOLT_REWRITE_MACHO_REWRITE_INSTANCE_H

#include "bolt/Utils/NameResolver.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {
class ToolOutputFile;
class RuntimeDyld;
class raw_pwrite_stream;
namespace object {
class MachOObjectFile;
} // namespace object

namespace bolt {

class BinaryContext;
class ProfileReaderBase;

class MachORewriteInstance {
  object::MachOObjectFile *InputFile;
  StringRef ToolPath;
  std::unique_ptr<BinaryContext> BC;

  NameResolver NR;

  std::unique_ptr<RuntimeDyld> RTDyld;

  std::unique_ptr<ToolOutputFile> Out;

  std::unique_ptr<ProfileReaderBase> ProfileReader;
  void preprocessProfileData();
  void processProfileDataPreCFG();
  void processProfileData();

  static StringRef getOrgSecPrefix() { return ".bolt.org"; }

  void mapInstrumentationSection(StringRef SectionName);
  void mapCodeSections();

  void adjustCommandLineOptions();
  void readSpecialSections();
  void discoverFileObjects();
  void disassembleFunctions();
  void buildFunctionsCFG();
  void postProcessFunctions();
  void runOptimizationPasses();
  void emitAndLink();

  void writeInstrumentationSection(StringRef SectionName,
                                   raw_pwrite_stream &OS);
  void rewriteFile();

public:
  // This constructor has complex initialization that can fail during
  // construction. Constructors can’t return errors, so clients must test \p Err
  // after the object is constructed. Use createMachORewriteInstance instead.
  MachORewriteInstance(object::MachOObjectFile *InputFile, StringRef ToolPath,
                       Error &Err);

  static Expected<std::unique_ptr<MachORewriteInstance>>
  createMachORewriteInstance(object::MachOObjectFile *InputFile,
                             StringRef ToolPath);
  ~MachORewriteInstance();

  Error setProfile(StringRef FileName);

  /// Run all the necessary steps to read, optimize and rewrite the binary.
  void run();
};

} // namespace bolt
} // namespace llvm

#endif
