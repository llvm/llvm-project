#include "llvm/Bitcode/ReaderWriter.h"
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

#define LLVM_360_AND_NEWER \
  (LLVM_VERSION_MAJOR > 3 || (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 6))

#define LLVM_350_AND_NEWER \
  (LLVM_VERSION_MAJOR > 3 || (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 5))

#if LLVM_350_AND_NEWER
#include <system_error>

#define ERROR_CODE std::error_code
#define UNIQUE_PTR std::unique_ptr
#else
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/system_error.h"

#define ERROR_CODE error_code
#define UNIQUE_PTR OwningPtr
#endif

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

int main(int argc, char **argv) {
  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "libclc builtin preparation tool\n");

  std::string ErrorMessage;
  std::auto_ptr<Module> M;

  {
#if LLVM_350_AND_NEWER
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputFilename);
    std::unique_ptr<MemoryBuffer> &BufferPtr = BufferOrErr.get();
    if (std::error_code  ec = BufferOrErr.getError())
#else
    UNIQUE_PTR<MemoryBuffer> BufferPtr;
    if (ERROR_CODE ec = MemoryBuffer::getFileOrSTDIN(InputFilename, BufferPtr))
#endif
      ErrorMessage = ec.message();
    else {
#if LLVM_VERSION_MAJOR > 3 || (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR > 4)
# if LLVM_360_AND_NEWER
      ErrorOr<Module *> ModuleOrErr =
	parseBitcodeFile(BufferPtr.get()->getMemBufferRef(), Context);
# else
      ErrorOr<Module *> ModuleOrErr = parseBitcodeFile(BufferPtr.get(), Context);
# endif
      if (ERROR_CODE ec = ModuleOrErr.getError())
        ErrorMessage = ec.message();
      M.reset(ModuleOrErr.get());
#else
      M.reset(ParseBitcodeFile(BufferPtr.get(), Context, &ErrorMessage));
#endif
    }
  }

  if (M.get() == 0) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "bitcode didn't read correctly.\n";
    return 1;
  }

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

#if LLVM_360_AND_NEWER
  std::error_code EC;
  UNIQUE_PTR<tool_output_file> Out
  (new tool_output_file(OutputFilename, EC, sys::fs::F_None));
  if (EC) {
    errs() << EC.message() << '\n';
    exit(1);
  }
#else
  std::string ErrorInfo;
  UNIQUE_PTR<tool_output_file> Out
  (new tool_output_file(OutputFilename.c_str(), ErrorInfo,
#if (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 4)
                        sys::fs::F_Binary));
#elif LLVM_VERSION_MAJOR > 3 || (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 5)
                        sys::fs::F_None));
#else
                        raw_fd_ostream::F_Binary));
#endif
  if (!ErrorInfo.empty()) {
    errs() << ErrorInfo << '\n';
    exit(1);
  }
#endif // LLVM_360_AND_NEWER

  WriteBitcodeToFile(M.get(), Out->os());

  // Declare success.
  Out->keep();
  return 0;
}

