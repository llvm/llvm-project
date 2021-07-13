/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#if !defined(__STDC_LIMIT_MACROS)
# define __STDC_LIMIT_MACROS
#endif
#if !defined(__STDC_CONSTANT_MACROS)
# define __STDC_CONSTANT_MACROS
#endif

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Config/llvm-config.h"

#include <system_error>

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

int main(int argc, char **argv) {
  LLVMContext Context;
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "bitcode library builtin preparation tool\n");

  std::string ErrorMessage;
  Module *M = nullptr;

  {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputFilename);
    if (std::error_code ec = BufferOrErr.getError())
      ErrorMessage = ec.message();
    else {
      std::unique_ptr<MemoryBuffer> &BufferPtr = BufferOrErr.get();
      Expected<std::unique_ptr<Module>> ModuleOrErr =
          parseBitcodeFile(BufferPtr.get()->getMemBufferRef(), Context);
      if (Error Err = ModuleOrErr.takeError()) {
        ErrorMessage = toString(std::move(Err));
      }
      else
        M = ModuleOrErr.get().release();
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

  // Set linkage of every external definition to linkonce_odr.
  for (Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage) {
        i->setLinkage(GlobalValue::LinkOnceODRLinkage);
    }
  }

  for (Module::global_iterator i = M->global_begin(), e = M->global_end();
       i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage) {
        i->setLinkage(GlobalValue::LinkOnceODRLinkage);
    }
  }

  for (Module::alias_iterator i = M->alias_begin(), e = M->alias_end();
       i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage) {
        i->setLinkage(GlobalValue::LinkOnceODRLinkage);
    }
  }


  if (OutputFilename.empty()) {
    errs() << "no output file\n";
    return 1;
  }

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> Out
  (new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
  if (EC) {
    errs() << EC.message() << '\n';
    exit(1);
  }

  WriteBitcodeToFile(*M, Out->os());

  // Declare success.
  Out->keep();
  return 0;
}

