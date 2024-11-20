#include "comgr-cache-command.h"

namespace COMGR {
using namespace llvm;
using namespace clang;

CachedCommand::CachedCommand(driver::Command &Command,
                             DiagnosticOptions &DiagOpts,
                             llvm::vfs::FileSystem &VFS,
                             ExecuteFnTy &&ExecuteImpl)
    : Command(Command), DiagOpts(DiagOpts), VFS(VFS),
      ExecuteImpl(std::move(ExecuteImpl)) {}

amd_comgr_status_t CachedCommand::execute(llvm::raw_ostream &LogS) {
  return ExecuteImpl(Command, LogS, DiagOpts, VFS);
}
} // namespace COMGR
