//===- lib/Config/Version.cpp - LLD Version Number ---------------*- C++-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines several version-related utility functions for LLD.
//
//===----------------------------------------------------------------------===//

#include "lld/Config/Version.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace lld {

StringRef getLLDRepositoryPath() {
#ifdef LLD_REPOSITORY_STRING
  return LLD_REPOSITORY_STRING;
#else
  return "";
#endif
}

StringRef getLLDRevision() {
#ifdef LLD_REVISION_STRING
  return LLD_REVISION_STRING;
#else
  return "";
#endif
}

std::string getLLDRepositoryVersion() {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
  std::string Path = getLLDRepositoryPath();
  std::string Revision = getLLDRevision();
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
  return OS.str();
}

StringRef getLLDVersion() {
#ifdef LLD_VERSION_STRING
  return LLD_VERSION_STRING;
#else
  return "";
#endif
}

} // end namespace lld
