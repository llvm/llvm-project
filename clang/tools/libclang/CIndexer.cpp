//===- CIndex.cpp - Clang-C Source Indexing Library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Clang-C Source Indexing library.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXString.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/YAMLParser.h"
#include <cstdio>

#ifdef __CYGWIN__
#include <cygwin/version.h>
#include <sys/cygwin.h>
#define LLVM_ON_WIN32 1
#endif

#ifdef LLVM_ON_WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace clang;

const std::string &CIndexer::getClangResourcesPath() {
  // Did we already compute the path?
  if (!ResourcesPath.empty())
    return ResourcesPath;

  SmallString<128> LibClangPath;

  // Find the location where this library lives (libclang.dylib).
#ifdef LLVM_ON_WIN32
  MEMORY_BASIC_INFORMATION mbi;
  char path[MAX_PATH];
  VirtualQuery((void *)(uintptr_t)clang_createTranslationUnit, &mbi,
               sizeof(mbi));
  GetModuleFileNameA((HINSTANCE)mbi.AllocationBase, path, MAX_PATH);

#ifdef __CYGWIN__
  char w32path[MAX_PATH];
  strcpy(w32path, path);
#if CYGWIN_VERSION_API_MAJOR > 0 || CYGWIN_VERSION_API_MINOR >= 181
  cygwin_conv_path(CCP_WIN_A_TO_POSIX, w32path, path, MAX_PATH);
#else
  cygwin_conv_to_full_posix_path(w32path, path);
#endif
#endif

  LibClangPath += llvm::sys::path::parent_path(path);
#else
  // This silly cast below avoids a C++ warning.
  Dl_info info;
  if (dladdr((void *)(uintptr_t)clang_createTranslationUnit, &info) == 0)
    llvm_unreachable("Call to dladdr() failed");

  // We now have the CIndex directory, locate clang relative to it.
  LibClangPath += llvm::sys::path::parent_path(info.dli_fname);
#endif

  llvm::sys::path::append(LibClangPath, "clang", CLANG_VERSION_STRING);

  // Cache our result.
  ResourcesPath = LibClangPath.str();
  return ResourcesPath;
}

StringRef CIndexer::getClangToolchainPath() {
  if (!ToolchainPath.empty())
    return ToolchainPath;
  StringRef ResourcePath = getClangResourcesPath();
  ToolchainPath = llvm::sys::path::parent_path(
      llvm::sys::path::parent_path(llvm::sys::path::parent_path(ResourcePath)));
  return ToolchainPath;
}

LibclangInvocationReporter::LibclangInvocationReporter(
    CIndexer &Idx, OperationKind Op, unsigned ParseOptions,
    llvm::ArrayRef<const char *> Args,
    llvm::ArrayRef<std::string> InvocationArgs,
    llvm::ArrayRef<CXUnsavedFile> UnsavedFiles) {
  StringRef Path = Idx.getInvocationEmissionPath();
  if (Path.empty())
    return;

  // Create a temporary file for the invocation log.
  SmallString<256> TempPath;
  TempPath = Path;
  llvm::sys::path::append(TempPath, "libclang-%%%%%%%%%%%%");
  int FD;
  if (llvm::sys::fs::createUniqueFile(TempPath, FD, TempPath))
    return;
  File = std::string(TempPath.begin(), TempPath.end());
  llvm::raw_fd_ostream OS(FD, /*ShouldClose=*/true);

  // Write out the information about the invocation to it.
  auto WriteStringKey = [&OS](StringRef Key, StringRef Value) {
    OS << R"(")" << Key << R"(":")";
    OS << llvm::yaml::escape(Value) << '"';
  };
  OS << '{';
  WriteStringKey("toolchain", Idx.getClangToolchainPath());
  OS << ',';
  WriteStringKey("libclang.operation",
                 Op == OperationKind::ParseOperation ? "parse" : "complete");
  OS << ',';
  OS << R"("libclang.opts":)" << ParseOptions;
  OS << ',';
  OS << R"("args":[)";
  for (const auto &I : llvm::enumerate(Args)) {
    if (I.index())
      OS << ',';
    OS << '"' << llvm::yaml::escape(I.value()) << '"';
  }
  if (!InvocationArgs.empty()) {
    OS << R"(],"invocation-args":[)";
    for (const auto &I : llvm::enumerate(InvocationArgs)) {
      if (I.index())
        OS << ',';
      OS << '"' << llvm::yaml::escape(I.value()) << '"';
    }
  }
  if (!UnsavedFiles.empty()) {
    OS << R"(],"unsaved_file_hashes":[)";
    for (const auto &UF : llvm::enumerate(UnsavedFiles)) {
      if (UF.index())
        OS << ',';
      OS << '{';
      WriteStringKey("name", UF.value().Filename);
      OS << ',';
      llvm::MD5 Hash;
      Hash.update(getContents(UF.value()));
      llvm::MD5::MD5Result Result;
      Hash.final(Result);
      SmallString<32> Digest = Result.digest();
      WriteStringKey("md5", Digest);
      OS << '}';
    }
  }
  OS << "]}";
}

LibclangInvocationReporter::~LibclangInvocationReporter() {
  if (!File.empty())
    llvm::sys::fs::remove(File);
}
