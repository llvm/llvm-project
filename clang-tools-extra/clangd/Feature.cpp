//===--- Feature.cpp - Compile-time configuration ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Feature.h"
#include "clang/Basic/Version.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Host.h"

namespace clang {
namespace clangd {

std::string versionString() { return clang::getClangToolFullVersion("clangd"); }

std::string platformString() {
  static std::string PlatformString = []() {
    std::string Host = llvm::sys::getProcessTriple();
    std::string Target = llvm::sys::getDefaultTargetTriple();
    if (Host != Target) {
      Host += "; target=";
      Host += Target;
    }
    return Host;
  }();
  return PlatformString;
}

std::string featureString() {
  return
#if defined(_WIN32)
      "windows"
#elif defined(__APPLE__)
      "mac"
#elif defined(__linux__)
      "linux"
#elif defined(LLVM_ON_UNIX)
      "unix"
#else
      "unknown"
#endif

#ifndef NDEBUG
      "+debug"
#endif
#if LLVM_ADDRESS_SANITIZER_BUILD
      "+asan"
#endif
#if LLVM_THREAD_SANITIZER_BUILD
      "+tsan"
#endif
#if LLVM_MEMORY_SANITIZER_BUILD
      "+msan"
#endif

#if CLANGD_ENABLE_REMOTE
      "+grpc"
#endif
#if CLANGD_BUILD_XPC
      "+xpc"
#endif

#if !CLANGD_TIDY_CHECKS
      "-tidy"
#endif

#if !CLANGD_DECISION_FOREST
      "-decision_forest"
#endif
      ;
}

} // namespace clangd
} // namespace clang
