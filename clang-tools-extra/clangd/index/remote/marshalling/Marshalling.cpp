//===--- Marshalling.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Marshalling.h"
#include "Logger.h"
#include "index/Serialization.h"

namespace clang {
namespace clangd {
namespace remote {

clangd::FuzzyFindRequest fromProtobuf(const FuzzyFindRequest *Request) {
  clangd::FuzzyFindRequest Result;
  Result.Query = Request->query();
  for (const auto &Scope : Request->scopes())
    Result.Scopes.push_back(Scope);
  Result.AnyScope = Request->any_scope();
  if (Request->limit())
    Result.Limit = Request->limit();
  Result.RestrictForCodeCompletion = Request->resricted_for_code_completion();
  for (const auto &Path : Request->proximity_paths())
    Result.ProximityPaths.push_back(Path);
  for (const auto &Type : Request->preferred_types())
    Result.ProximityPaths.push_back(Type);
  return Result;
}

llvm::Optional<clangd::Symbol> fromProtobuf(const Symbol &Message,
                                            llvm::UniqueStringSaver *Strings) {
  auto Result = symbolFromYAML(Message.yaml_serialization(), Strings);
  if (!Result) {
    elog("Cannot convert Symbol from Protobuf: {}", Result.takeError());
    return llvm::None;
  }
  return *Result;
}
llvm::Optional<clangd::Ref> fromProtobuf(const Ref &Message,
                                         llvm::UniqueStringSaver *Strings) {
  auto Result = refFromYAML(Message.yaml_serialization(), Strings);
  if (!Result) {
    elog("Cannot convert Ref from Protobuf: {}", Result.takeError());
    return llvm::None;
  }
  return *Result;
}

LookupRequest toProtobuf(const clangd::LookupRequest &From) {
  LookupRequest RPCRequest;
  for (const auto &SymbolID : From.IDs)
    RPCRequest.add_ids(SymbolID.str());
  return RPCRequest;
}

FuzzyFindRequest toProtobuf(const clangd::FuzzyFindRequest &From) {
  FuzzyFindRequest RPCRequest;
  RPCRequest.set_query(From.Query);
  for (const auto &Scope : From.Scopes)
    RPCRequest.add_scopes(Scope);
  RPCRequest.set_any_scope(From.AnyScope);
  if (From.Limit)
    RPCRequest.set_limit(*From.Limit);
  RPCRequest.set_resricted_for_code_completion(From.RestrictForCodeCompletion);
  for (const auto &Path : From.ProximityPaths)
    RPCRequest.add_proximity_paths(Path);
  for (const auto &Type : From.PreferredTypes)
    RPCRequest.add_preferred_types(Type);
  return RPCRequest;
}

RefsRequest toProtobuf(const clangd::RefsRequest &From) {
  RefsRequest RPCRequest;
  for (const auto &ID : From.IDs)
    RPCRequest.add_ids(ID.str());
  RPCRequest.set_filter(static_cast<uint32_t>(From.Filter));
  if (From.Limit)
    RPCRequest.set_limit(*From.Limit);
  return RPCRequest;
}

Symbol toProtobuf(const clangd::Symbol &From) {
  Symbol Result;
  Result.set_yaml_serialization(toYAML(From));
  return Result;
}

Ref toProtobuf(const clangd::Ref &From) {
  Ref Result;
  Result.set_yaml_serialization(toYAML(From));
  return Result;
}

} // namespace remote
} // namespace clangd
} // namespace clang
