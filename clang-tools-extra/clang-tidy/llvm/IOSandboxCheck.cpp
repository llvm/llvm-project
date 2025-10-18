//===--- FilesystemAccessCheck.cpp - clang-tidy --------------------------===//
//
// Enforces controlled filesystem access patterns
//
//===----------------------------------------------------------------------===//

#include "IOSandboxCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {
// Low-level filesystem functions that should only be called from llvm::sys::fs
static const llvm::StringSet<> &getForbiddenFilesystemFunctions() {
  static const llvm::StringSet<> Functions = {
      // POSIX file operations
      "open",
      "openat",
      "creat",
      "close",
      "read",
      "write",
      "pread",
      "pwrite",
      "lseek",
      "ftruncate",
      "truncate",
      "stat",
      "fstat",
      "lstat",
      "fstatat",
      "access",
      "faccessat",
      "chmod",
      "fchmod",
      "fchmodat",
      "chown",
      "fchown",
      "lchown",
      "fchownat",
      "link",
      "linkat",
      "symlink",
      "symlinkat",
      "readlink",
      "readlinkat",
      "unlink",
      "unlinkat",
      "remove",
      "rename",
      "renameat",
      "mkdir",
      "mkdirat",
      "rmdir",
      "opendir",
      "readdir",
      "closedir",
      "fdopendir",
      "chdir",
      "fchdir",
      "getcwd",
      "dup",
      "dup2",
      "dup3",
      "fcntl",
      "pipe",
      "pipe2",
      "mkfifo",
      "mkfifoat",
      "mknod",
      "mknodat",
      "utimes",
      "futimes",
      "utimensat",
      "futimens",

      // C standard library file operations
      "fopen",
      "freopen",
      "fclose",
      "fflush",
      "fread",
      "fwrite",
      "fgetc",
      "fputc",
      "fgets",
      "fputs",
      "fseek",
      "ftell",
      "rewind",
      "fgetpos",
      "fsetpos",
      "tmpfile",
      "tmpnam",
      "tempnam",

      // Windows file operations
      "CreateFileA",
      "CreateFileW",
      "CreateFile",
      "ReadFile",
      "WriteFile",
      "CloseHandle",
      "DeleteFileA",
      "DeleteFileW",
      "DeleteFile",
      "MoveFileA",
      "MoveFileW",
      "MoveFile",
      "CopyFileA",
      "CopyFileW",
      "CopyFile",
      "GetFileAttributesA",
      "GetFileAttributesW",
      "GetFileAttributes",
      "SetFileAttributesA",
      "SetFileAttributesW",
      "SetFileAttributes",
      "CreateDirectoryA",
      "CreateDirectoryW",
      "CreateDirectory",
      "RemoveDirectoryA",
      "RemoveDirectoryW",
      "RemoveDirectory",
      "FindFirstFileA",
      "FindFirstFileW",
      "FindFirstFile",
      "FindNextFileA",
      "FindNextFileW",
      "FindNextFile",
      "FindClose",
      "GetCurrentDirectoryA",
      "GetCurrentDirectoryW",
      "SetCurrentDirectoryA",
      "SetCurrentDirectoryW",

      // Memory-mapped files
      "mmap",
      "munmap",
      "mprotect",
      "msync",
      "MapViewOfFile",
      "UnmapViewOfFile",
  };
  return Functions;
}

static bool isInLLVMSysFsNamespace(const FunctionDecl *FD) {
  if (!FD)
    return false;

  auto IsAnonymousNamespace = [](const DeclContext *DC) {
    if (!DC)
      return false;
    const auto *ND = dyn_cast<NamespaceDecl>(DC);
    if (!ND)
      return false;
    return ND->isAnonymousNamespace();
  };

  auto GetNamedNamespace = [](const DeclContext *DC) -> const NamespaceDecl * {
    if (!DC)
      return nullptr;
    const auto *ND = dyn_cast<NamespaceDecl>(DC);
    if (!ND)
      return nullptr;
    if (ND->isAnonymousNamespace())
      return nullptr;
    return ND;
  };

  const DeclContext *DC = FD->getDeclContext();

  // Walk up the context chain looking for llvm::sys::fs
  SmallVector<StringRef> ReverseNamespaces;
  while (IsAnonymousNamespace(DC))
    DC = DC->getParent();
  while (const auto *ND = GetNamedNamespace(DC)) {
    ReverseNamespaces.push_back(ND->getName());
    DC = DC->getParent();
  }
  auto Namespaces = llvm::reverse(ReverseNamespaces);

  return llvm::equal(Namespaces, SmallVector<StringRef>{"llvm", "sys", "fs"});
}

static bool isLLVMSysFsCall(const CallExpr *CE) {
  if (!CE)
    return false;

  const FunctionDecl *Callee = CE->getDirectCallee();
  if (!Callee)
    return false;

  return isInLLVMSysFsNamespace(Callee) && !Callee->isOverloadedOperator();
}

static bool isForbiddenFilesystemCall(const CallExpr *CE) {
  if (!CE)
    return false;

  const FunctionDecl *Callee = CE->getDirectCallee();
  if (!Callee)
    return false;

  const auto &ForbiddenFuncs = getForbiddenFilesystemFunctions();

  return ForbiddenFuncs.contains(Callee->getQualifiedNameAsString());
}

static bool hasSandboxBypass(const FunctionDecl *FD, SourceLocation CallLoc) {
  if (!FD || !FD->hasBody())
    return false;

  const Stmt *Body = FD->getBody();
  if (!Body)
    return false;

  // Look for variable declarations of the bypass type
  // We need to check if the bypass variable is declared before the call site
  class BypassFinder : public RecursiveASTVisitor<BypassFinder> {
  public:
    bool FoundBypass = false;
    SourceLocation CallLocation;
    const SourceManager *SM;

    bool VisitVarDecl(VarDecl *VD) {
      if (!VD)
        return true;

      // Check if this is a sandbox bypass variable
      const Type *T = VD->getType().getTypePtrOrNull();
      if (!T)
        return true;

      const CXXRecordDecl *RD = T->getAsCXXRecordDecl();
      if (!RD)
        return true;

      // Check for ScopedSandboxDisable or similar RAII types
      std::string TypeName = RD->getQualifiedNameAsString();
      if (TypeName.find("ScopedSandboxDisable") != std::string::npos ||
          TypeName.find("scopedDisable") != std::string::npos) {

        // Check if this declaration comes before the call
        if (SM &&
            SM->isBeforeInTranslationUnit(VD->getLocation(), CallLocation)) {
          FoundBypass = true;
          return false; // Stop searching
        }
      }

      return true;
    }
  };

  BypassFinder Finder;
  Finder.CallLocation = CallLoc;
  Finder.SM = &FD->getASTContext().getSourceManager();
  Finder.TraverseStmt(const_cast<Stmt *>(Body));

  return Finder.FoundBypass;
}

void IOSandboxCheck::registerMatchers(MatchFinder *Finder) {
  // Match any call expression within a function.
  Finder->addMatcher(
      callExpr(hasAncestor(functionDecl().bind("parent_func"))).bind("call"),
      this);

  // Also match variable declarations to find sandbox bypass objects
  Finder->addMatcher(
      varDecl(hasType(cxxRecordDecl(hasName("ScopedSandboxDisable"))),
              hasAncestor(functionDecl().bind("func_with_bypass")))
          .bind("bypass_var"),
      this);
}

void IOSandboxCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  const auto *ParentFunc = Result.Nodes.getNodeAs<FunctionDecl>("parent_func");

  if (!Call || !ParentFunc)
    return;

  // Skip system headers and template instantiations
  if (Call->getBeginLoc().isInvalid() ||
      Result.Context->getSourceManager().isInSystemHeader(
          Call->getBeginLoc()) ||
      ParentFunc->isTemplateInstantiation())
    return;

  // Rule 1: Check if calling llvm::sys::fs without sandbox bypass
  if (isLLVMSysFsCall(Call)) {
    if (!hasSandboxBypass(ParentFunc, Call->getBeginLoc())) {
      diag(Call->getBeginLoc(), "call to llvm::sys::fs function")
          << Call->getSourceRange();
    }
    return; // Don't check rule 2 for llvm::sys::fs calls
  }

  // Rule 2: Check if calling forbidden filesystem functions outside
  // llvm::sys::fs
  if (isForbiddenFilesystemCall(Call)) {
    if (!isInLLVMSysFsNamespace(ParentFunc)) {
      const auto *Callee = Call->getDirectCallee();
      std::string CalleeName = Callee ? Callee->getNameAsString() : "unknown";

      diag(Call->getBeginLoc(),
           "low-level filesystem function '%0' may only be called from a "
           "llvm::sys::fs function")
          << CalleeName << Call->getSourceRange();
    }
  }
}
} // namespace clang::tidy::llvm_check
