#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

int main(int argc, char *argv[]) {
  using namespace llvm;
  using namespace llvm::sys;

  StringRef Executable = argv[0];
  StringRef Alias = sys::path::filename(Executable);

  llvm::ExitOnError Exit((Alias + ": ").str());

  if (!Alias.consume_front("amd")) {
    Exit(createStringError("binary '" + Alias + "' not prefixed by 'amd'."));
  }

  void *MainAddr = reinterpret_cast<void *>(main);
  std::string AMDLlvmPath = fs::getMainExecutable(argv[0], MainAddr);
  if (AMDLlvmPath.empty()) {
    Exit(createStringError(
        "couldn't figure out path to LLVM install bin/ directory."));
  }

  StringRef BinaryDir = path::parent_path(AMDLlvmPath);

  SmallString<256> BinaryPath;
  sys::path::append(BinaryPath, BinaryDir, Alias);

  if (!fs::exists(BinaryPath)) {
    Exit(createStringError("binary '" + BinaryPath + "' does not exist."));
  }

  SmallVector<StringRef, 128> Argv = {BinaryPath};
  Argv.insert(Argv.end(), argv + 1, argv + argc);

  return ExecuteAndWait(BinaryPath, Argv);
}
