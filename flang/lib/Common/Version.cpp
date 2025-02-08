//===- Version.cpp - Flang Version Number -------------------*- Fortran -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several version-related utility functions for Flang.
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Version.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>

#include "VCSVersion.inc"

namespace Fortran::common {

std::string getFlangRepositoryPath() {
#if defined(FLANG_REPOSITORY_STRING)
  return FLANG_REPOSITORY_STRING;
#else
#ifdef FLANG_REPOSITORY
  return FLANG_REPOSITORY;
#else
  return "";
#endif
#endif
}

std::string getLLVMRepositoryPath() {
#ifdef LLVM_REPOSITORY
  return LLVM_REPOSITORY;
#else
  return "";
#endif
}

std::string getFlangRevision() {
#ifdef FLANG_REVISION
  return FLANG_REVISION;
#else
  return "";
#endif
}

std::string getLLVMRevision() {
#ifdef LLVM_REVISION
  return LLVM_REVISION;
#else
  return "";
#endif
}

std::string getFlangFullRepositoryVersion() {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
  std::string Path = getFlangRepositoryPath();
  std::string Revision = getFlangRevision();
  if (!Path.empty() || !Revision.empty()) {
    OS << '(';
    if (!Path.empty())
      OS << Path;
    if (!Revision.empty()) {
      if (!Path.empty())
        OS << ' ';
      OS << Revision;
    }
    OS << ')';
  }
  // Support LLVM in a separate repository.
  std::string LLVMRev = getLLVMRevision();
  if (!LLVMRev.empty() && LLVMRev != Revision) {
    OS << " (";
    std::string LLVMRepo = getLLVMRepositoryPath();
    if (!LLVMRepo.empty())
      OS << LLVMRepo << ' ';
    OS << LLVMRev << ')';
  }
  return buf;
}

std::string getFlangFullVersion() { return getFlangToolFullVersion("flang"); }

std::string getFlangToolFullVersion(llvm::StringRef ToolName) {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
#ifdef FLANG_VENDOR
  OS << FLANG_VENDOR;
#endif
  OS << ToolName << " version " FLANG_VERSION_STRING;

  std::string repo = getFlangFullRepositoryVersion();
  if (!repo.empty()) {
    OS << " " << repo;
  }

  return buf;
}

} // end namespace Fortran::common
