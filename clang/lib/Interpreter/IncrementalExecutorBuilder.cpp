//===-- IncrementalExecutorBuilder.cpp - Executor Builder ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the common incremental executor builder.
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/IncrementalExecutor.h"

namespace clang {

IncrementalExecutorBuilder::~IncrementalExecutorBuilder() = default;

llvm::Expected<std::unique_ptr<IncrementalExecutor>>
IncrementalExecutorBuilder::create(llvm::orc::ThreadSafeContext &TSC,
                                   const clang::TargetInfo &TI) {
  if (IE)
    return std::move(IE);
  return createExecutor(TSC, TI);
}

} // namespace clang
