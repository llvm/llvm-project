//===- comgr-cache-command.h - CacheCommand implementation ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_CACHE_COMMAND_H
#define COMGR_CACHE_COMMAND_H

#include "amd_comgr.h"

#include <clang/Driver/Action.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>

namespace llvm {
class raw_ostream;
}

namespace COMGR {
class CachedCommandAdaptor {
public:
  using ActionClass =
      std::underlying_type_t<clang::driver::Action::ActionClass>;
  using HashAlgorithm = llvm::SHA256;
  using Identifier = llvm::SmallString<64>;

  llvm::Expected<Identifier> getIdentifier() const;

  virtual bool canCache() const = 0;
  virtual llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) = 0;
  virtual llvm::Expected<llvm::StringRef> readExecuteOutput() = 0;
  virtual amd_comgr_status_t execute(llvm::raw_ostream &LogS) = 0;

  virtual ~CachedCommandAdaptor() = default;

  // helper to work around the comgr-xxxxx string appearing in files
  static void addFileContents(HashAlgorithm &H, llvm::StringRef Buf);
  static void addUInt(HashAlgorithm &H, uint64_t I);
  static void addString(HashAlgorithm &H, llvm::StringRef S);

  struct ComgrTmpSearchResult {
    size_t StartPosition;
    size_t MatchSize;
  };
  static std::optional<ComgrTmpSearchResult>
  searchComgrTmpModel(llvm::StringRef S);

  // helper since several command types just write to a single output file
  static llvm::Error writeSingleOutputFile(llvm::StringRef OutputFilename,
                                           llvm::StringRef CachedBuffer);
  static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  readSingleOutputFile(llvm::StringRef OutputFilename);

protected:
  virtual ActionClass getClass() const = 0;
  virtual void addOptionsIdentifier(HashAlgorithm &) const = 0;
  virtual llvm::Error addInputIdentifier(HashAlgorithm &) const = 0;
};
} // namespace COMGR

#endif
