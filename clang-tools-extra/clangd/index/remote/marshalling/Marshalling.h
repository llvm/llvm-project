//===--- Marshalling.h -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transformations between native Clangd types and Protobuf-generated classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_MARSHALLING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_MARSHALLING_H

#include "Index.grpc.pb.h"
#include "index/Index.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace clangd {
namespace remote {

clangd::FuzzyFindRequest fromProtobuf(const FuzzyFindRequest *Request);
llvm::Optional<clangd::Symbol> fromProtobuf(const Symbol &Message,
                                            llvm::UniqueStringSaver *Strings);
llvm::Optional<clangd::Ref> fromProtobuf(const Ref &Message,
                                         llvm::UniqueStringSaver *Strings);

LookupRequest toProtobuf(const clangd::LookupRequest &From);
FuzzyFindRequest toProtobuf(const clangd::FuzzyFindRequest &From);
RefsRequest toProtobuf(const clangd::RefsRequest &From);

Ref toProtobuf(const clangd::Ref &From);
Symbol toProtobuf(const clangd::Symbol &From);

} // namespace remote
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_MARSHALLING_H
