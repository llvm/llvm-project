//===- bolt/Rewrite/JITRewriteInstance.h - in-memory rewriter ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to control BOLT as JIT library
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_JIT_REWRITE_REWRITE_INSTANCE_H
#define BOLT_JIT_REWRITE_REWRITE_INSTANCE_H

#include "bolt/Utils/NameResolver.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"
#include <memory>

namespace llvm {

namespace object {
class ObjectFile;
}

namespace bolt {

class BinaryContext;
class ProfileReaderBase;
struct JournalingStreams;

/// Allows a process to instrospect itself by running BOLT to disassemble its
/// its own address space.
class JITRewriteInstance {
  std::unique_ptr<BinaryContext> BC;
  NameResolver NR;
  StringSaver StrPool;
  BumpPtrAllocator StrAllocator;
  std::unique_ptr<ProfileReaderBase> ProfileReader;

  void adjustCommandLineOptions();
  Error preprocessProfileData();
  Error processProfileDataPreCFG();
  Error processProfileData();
  Error disassembleFunctions();
  Error buildFunctionsCFG();
  void postProcessFunctions();
  JITRewriteInstance(JournalingStreams Logger, bool IsPIC, Error &Err);

public:
  /// Create BOLT data structures/interface to deal with disassembly. Logger
  /// contains the streams used for BOLT to report events (regular or errors)
  /// that might happen while BOLT is trying to reconstruct a function from
  /// binary level.
  static Expected<std::unique_ptr<JITRewriteInstance>>
  createJITRewriteInstance(JournalingStreams Logger, bool IsPIC);
  ~JITRewriteInstance();

  /// This is the main entry point used to make BOLT aware of a fragment of
  /// memory space in the process. The user might need to reconstruct the
  /// original ELF type/flags, such as using SHT_PROGBITS to inform
  /// this is allocatable region and flags SHF_ALLOC | SHF_EXECINSTR to
  /// flag a section containing code.
  void registerJITSection(StringRef Name, uint64_t Address, StringRef Data,
                          unsigned Alignment, unsigned ELFType,
                          unsigned ELFFlags);

  /// Communicate to BOLT the boundaries of a function in a section of memory
  /// previously registered with registerJITSection.
  void registerJITFunction(StringRef Name, uintptr_t Addr, size_t Size);

  /// In case the user is using LLVM as an in-process JIT, and the user has
  /// access over the ObjectFile instance loaded in memory, instead of using
  /// registerJITSection/registerJITFunction pair, the user can just forward
  /// that object here and JITRewriteInstance will read this object and call
  /// registerJITSection/registerJITFunction the appropriate number of times
  /// to map this object to BOLT.
  Error notifyObjectLoaded(const object::ObjectFile &Obj);

  /// Mark all functions added so far as non-simple, so BOLT will skip them.
  void disableAllFunctions();

  /// Mark an specific function as simple, so BOLT will try to disassemble it.
  void processFunctionContaining(uint64_t Address);

  /// Supply a profile file for BOLT to attach edge counts to the disassembled
  /// functions.
  Error setProfile(StringRef FileName);

  /// Run all the necessary steps to disassemble registered sections and
  /// functions (process what we have so far).
  Error run();

  /// Print all BOLT's processed functions
  void printAll(raw_ostream &OS);

  /// Print a specific function processed by BOLT
  void printFunctionContaining(raw_ostream &OS, uint64_t Address);
};

} // namespace bolt
} // namespace llvm

#endif
