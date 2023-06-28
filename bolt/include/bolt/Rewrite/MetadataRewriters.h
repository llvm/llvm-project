//===- bolt/Rewrite/MetadataRewriters.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_METADATA_REWRITERS_H
#define BOLT_REWRITE_METADATA_REWRITERS_H

#include <memory>

namespace llvm {
namespace bolt {

class MetadataRewriter;

/// The list of rewriter build functions (e.g. createNewMetadataRewriter()) that
/// return std::unique_ptr<MetadataRewriter>.

} // namespace bolt
} // namespace llvm

#endif // BOLT_REWRITE_METADATA_REWRITERS_H
