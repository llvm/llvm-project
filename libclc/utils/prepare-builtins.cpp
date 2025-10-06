//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

using namespace llvm;

static ExitOnError ExitOnErr;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

static cl::opt<bool> TextualOut("S", cl::desc("Emit LLVM textual assembly"),
                                cl::init(false));

int main(int argc, char **argv) {
  LLVMContext Context;
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "libclc builtin preparation tool\n");

  std::string ErrorMessage;
  Module *M = nullptr;

  {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputFilename);
    if (std::error_code  ec = BufferOrErr.getError()) {
      ErrorMessage = ec.message();
    } else {
      std::unique_ptr<MemoryBuffer> &BufferPtr = BufferOrErr.get();
      SMDiagnostic Err;
      std::unique_ptr<llvm::Module> MPtr =
          ExitOnErr(Expected<std::unique_ptr<llvm::Module>>(
              parseIR(BufferPtr.get()->getMemBufferRef(), Err, Context)));
      M = MPtr.release();
    }
  }

  if (!M) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "bitcode didn't read correctly.\n";
    return 1;
  }

  // Strip the OpenCL version metadata. There are a lot of linked
  // modules in the library build, each spamming the same
  // version. This may also report a different version than the user
  // program is using. This should probably be uniqued when linking.
  if (NamedMDNode *OCLVersion = M->getNamedMetadata("opencl.ocl.version"))
      M->eraseNamedMetadata(OCLVersion);

  // Set linkage of every external definition to linkonce_odr.
  for (Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage)
      i->setLinkage(GlobalValue::LinkOnceODRLinkage);
  }

  for (Module::global_iterator i = M->global_begin(), e = M->global_end();
       i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage)
      i->setLinkage(GlobalValue::LinkOnceODRLinkage);
  }

  if (OutputFilename.empty()) {
    errs() << "no output file\n";
    return 1;
  }

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> Out(
      new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
  if (EC) {
    errs() << EC.message() << '\n';
    exit(1);
  }

  if (TextualOut)
    M->print(Out->os(), nullptr, true);
  else
    WriteBitcodeToFile(*M, Out->os());

  // Declare success.
  Out->keep();
  return 0;
}
