//===- llvm-bolt-fuzzer.cpp - Fuzzing target for llvm-bolt ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace opts {
extern cl::opt<std::string> OutputFilename;
extern cl::opt<bool> Lite;
} // namespace opts

extern "C" int LLVMFuzzerTestOneInput(const char *Data, size_t Size) {
  const char *argv[] = {"llvm-bolt", nullptr};
  const char argc = 1;
  opts::OutputFilename = "/dev/null";
  opts::Lite = false;

  // Input has to be an ELF - we don't want to fuzz createBinary interface.
  if (Size < 4 || strncmp("\177ELF", Data, 4) != 0)
    return 0;
  // Construct an ELF binary from fuzzer input.
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(StringRef(Data, Size), "", false);
  Expected<std::unique_ptr<Binary>> BinaryOrErr =
      createBinary(Buffer->getMemBufferRef());
  // Check that the input is a valid binary.
  if (Error E = BinaryOrErr.takeError()) {
    consumeError(std::move(E));
    return 0;
  }
  Binary &Binary = *BinaryOrErr.get();
  // Check that the binary is an ELF64LE object file.
  auto *E = dyn_cast<ELF64LEObjectFile>(&Binary);
  if (!E)
    return 0;

  // Fuzz RewriteInstance.
  auto RIOrErr =
      RewriteInstance::createRewriteInstance(E, argc, argv, "llvm-bolt");
  if (Error E = RIOrErr.takeError()) {
    consumeError(std::move(E));
    return 0;
  }
  RewriteInstance &RI = *RIOrErr.get();
  RI.run();
  return 0;
}

extern "C" LLVM_ATTRIBUTE_USED int LLVMFuzzerInitialize(int *argc,
                                                        char ***argv) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  return 0;
}
