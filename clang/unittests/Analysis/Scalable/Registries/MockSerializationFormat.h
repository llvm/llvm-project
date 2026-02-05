//===- MockSerializationFormat.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_REGISTRIES_MOCKSERIALIZATIONFORMAT_H
#define LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_REGISTRIES_MOCKSERIALIZATIONFORMAT_H

#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"

namespace clang::ssaf {

class MockSerializationFormat final : public SerializationFormat {
public:
  explicit MockSerializationFormat(
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  TUSummary readTUSummary(llvm::StringRef Path) override;

  void writeTUSummary(const TUSummary &Summary,
                      llvm::StringRef OutputDir) override;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_REGISTRIES_MOCKSERIALIZATIONFORMAT_H
