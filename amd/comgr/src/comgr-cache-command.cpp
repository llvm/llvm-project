//===- comgr-cache-command.cpp - CacheCommand implementation --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CachedCommandAdaptor: the interface and common
/// operations for commands that save their execution results in the cache.
///
//===----------------------------------------------------------------------===//

#include "comgr-cache-command.h"
#include "comgr-cache.h"
#include "comgr-device-libs.h"
#include "comgr-env.h"
#include "comgr.h"

#include <clang/Basic/Version.h>
#include <llvm/ADT/StringExtras.h>

#include <optional>

namespace COMGR {
using namespace llvm;
using namespace clang;

std::optional<CachedCommandAdaptor::ComgrTmpSearchResult>
CachedCommandAdaptor::searchComgrTmpModel(StringRef S) {
  // Ideally, we would use std::regex_search with the regex
  // "comgr-[[:num:]]+-[[:num:]]+-[[:alnum:]]{6}". However, due to a bug in
  // stdlibc++ (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85824) we have to
  // roll our own search of this regular expression. This bug resulted in a
  // crash in luxmarkv3, during the std::regex constructor.

  const StringRef Prefix = "comgr";
  const size_t AlnumCount = 6;

  StringRef Remaining = S;
  while (!Remaining.empty()) {
    size_t PosInRemaining = Remaining.find(Prefix);
    if (PosInRemaining == StringRef::npos)
      return std::nullopt;

    size_t PosInS = Remaining.data() + PosInRemaining - S.data();

    Remaining = Remaining.substr(PosInRemaining + Prefix.size());

    unsigned Pid;
    if (!Remaining.consume_front("-") ||
        Remaining.consumeInteger<unsigned>(10, Pid)) {
      continue;
    }

    unsigned Id;
    if (!Remaining.consume_front("-") ||
        Remaining.consumeInteger<unsigned>(10, Id)) {
      continue;
    }

    if (!Remaining.consume_front("-")) {
      continue;
    }

    if (Remaining.size() < AlnumCount) {
      continue;
    }

    // Use llvm::isAlnum and not std::isalnum. The later is locale dependent and
    // can have issues depending on the stdlib version and application.
    if (!all_of(Remaining.substr(0, AlnumCount), llvm::isAlnum)) {
      continue;
    }

    // `Remaining` begin is one after the end of the pattern
    Remaining = Remaining.drop_front(AlnumCount);

    size_t MatchSize = Remaining.data() - S.data() - PosInS;

    return {{PosInS, MatchSize}};
  }

  return std::nullopt;
}

void CachedCommandAdaptor::addUInt(CachedCommandAdaptor::HashAlgorithm &H,
                                   uint64_t I) {
  uint8_t Bytes[sizeof(I)];
  memcpy(&Bytes, &I, sizeof(I));
  H.update(Bytes);
}

void CachedCommandAdaptor::addString(CachedCommandAdaptor::HashAlgorithm &H,
                                     StringRef S) {
  // hash size + contents to avoid collisions
  // for example, we have to ensure that the result of hashing "AA" "BB" is
  // different from "A" "ABB"
  addUInt(H, S.size());
  H.update(S);
}

void CachedCommandAdaptor::addFileContents(
    CachedCommandAdaptor::HashAlgorithm &H, StringRef Buf) {
  // this is a workaround temporary paths getting in the output files of the
  // different commands in #line directives in preprocessed files, and the
  // ModuleID or source_filename in the bitcode.
  while (!Buf.empty()) {
    auto ComgrTmpPos = searchComgrTmpModel(Buf);
    if (!ComgrTmpPos) {
      addString(H, Buf);
      break;
    }

    StringRef ToHash = Buf.substr(0, ComgrTmpPos->StartPosition);
    addString(H, ToHash);
    Buf = Buf.substr(ToHash.size() + ComgrTmpPos->MatchSize);
  }
}

Expected<CachedCommandAdaptor::Identifier>
CachedCommandAdaptor::getIdentifier() const {
  CachedCommandAdaptor::HashAlgorithm H;
  H.update(getClass());
  H.update(env::shouldEmitVerboseLogs());
  addString(H, getClangFullVersion());
  addString(H, getComgrHashIdentifier());
  H.update(getDeviceLibrariesIdentifier());

  if (Error E = addInputIdentifier(H))
    return E;

  addOptionsIdentifier(H);

  CachedCommandAdaptor::Identifier Id;
  toHex(H.final(), true, Id);
  return Id;
}

llvm::Error
CachedCommandAdaptor::writeSingleOutputFile(StringRef OutputFilename,
                                            StringRef CachedBuffer) {
  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC);
  if (EC) {
    Error E = createStringError(EC, Twine("Failed to open ") + OutputFilename +
                                        " : " + EC.message() + "\n");
    return E;
  }

  Out.write(CachedBuffer.data(), CachedBuffer.size());
  Out.close();
  if (Out.has_error()) {
    Error E = createStringError(EC, Twine("Failed to write ") + OutputFilename +
                                        " : " + EC.message() + "\n");
    return E;
  }

  return Error::success();
}

Expected<std::unique_ptr<MemoryBuffer>>
CachedCommandAdaptor::readSingleOutputFile(StringRef OutputFilename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
      MemoryBuffer::getFile(OutputFilename);
  if (!MBOrErr) {
    std::error_code EC = MBOrErr.getError();
    return createStringError(EC, Twine("Failed to open ") + OutputFilename +
                                     " : " + EC.message() + "\n");
  }

  return std::move(*MBOrErr);
}
} // namespace COMGR
