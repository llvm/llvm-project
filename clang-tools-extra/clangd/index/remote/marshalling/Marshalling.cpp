//===--- Marshalling.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Marshalling.h"
#include "Index.pb.h"
#include "Protocol.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "index/SymbolLocation.h"
#include "index/SymbolOrigin.h"
#include "support/Logger.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace clangd {
namespace remote {

namespace {

clangd::SymbolLocation::Position fromProtobuf(const Position &Message) {
  clangd::SymbolLocation::Position Result;
  Result.setColumn(static_cast<uint32_t>(Message.column()));
  Result.setLine(static_cast<uint32_t>(Message.line()));
  return Result;
}

Position toProtobuf(const clangd::SymbolLocation::Position &Position) {
  remote::Position Result;
  Result.set_column(Position.column());
  Result.set_line(Position.line());
  return Result;
}

clang::index::SymbolInfo fromProtobuf(const SymbolInfo &Message) {
  clang::index::SymbolInfo Result;
  Result.Kind = static_cast<clang::index::SymbolKind>(Message.kind());
  Result.SubKind = static_cast<clang::index::SymbolSubKind>(Message.subkind());
  Result.Lang = static_cast<clang::index::SymbolLanguage>(Message.language());
  Result.Properties =
      static_cast<clang::index::SymbolPropertySet>(Message.properties());
  return Result;
}

SymbolInfo toProtobuf(const clang::index::SymbolInfo &Info) {
  SymbolInfo Result;
  Result.set_kind(static_cast<uint32_t>(Info.Kind));
  Result.set_subkind(static_cast<uint32_t>(Info.SubKind));
  Result.set_language(static_cast<uint32_t>(Info.Lang));
  Result.set_properties(static_cast<uint32_t>(Info.Properties));
  return Result;
}

clangd::SymbolLocation fromProtobuf(const SymbolLocation &Message,
                                    llvm::UniqueStringSaver *Strings) {
  clangd::SymbolLocation Location;
  Location.Start = fromProtobuf(Message.start());
  Location.End = fromProtobuf(Message.end());
  Location.FileURI = Strings->save(Message.file_uri()).begin();
  return Location;
}

SymbolLocation toProtobuf(const clangd::SymbolLocation &Location) {
  remote::SymbolLocation Result;
  *Result.mutable_start() = toProtobuf(Location.Start);
  *Result.mutable_end() = toProtobuf(Location.End);
  *Result.mutable_file_uri() = Location.FileURI;
  return Result;
}

HeaderWithReferences
toProtobuf(const clangd::Symbol::IncludeHeaderWithReferences &IncludeHeader) {
  HeaderWithReferences Result;
  Result.set_header(IncludeHeader.IncludeHeader.str());
  Result.set_references(IncludeHeader.References);
  return Result;
}

clangd::Symbol::IncludeHeaderWithReferences
fromProtobuf(const HeaderWithReferences &Message) {
  return clangd::Symbol::IncludeHeaderWithReferences{Message.header(),
                                                     Message.references()};
}

} // namespace

clangd::FuzzyFindRequest fromProtobuf(const FuzzyFindRequest *Request) {
  clangd::FuzzyFindRequest Result;
  Result.Query = Request->query();
  for (const auto &Scope : Request->scopes())
    Result.Scopes.push_back(Scope);
  Result.AnyScope = Request->any_scope();
  if (Request->limit())
    Result.Limit = Request->limit();
  Result.RestrictForCodeCompletion = Request->restricted_for_code_completion();
  for (const auto &Path : Request->proximity_paths())
    Result.ProximityPaths.push_back(Path);
  for (const auto &Type : Request->preferred_types())
    Result.ProximityPaths.push_back(Type);
  return Result;
}

llvm::Optional<clangd::Symbol> fromProtobuf(const Symbol &Message,
                                            llvm::UniqueStringSaver *Strings) {
  if (!Message.has_info() || !Message.has_definition() ||
      !Message.has_canonical_declarattion()) {
    elog("Cannot convert Symbol from Protobuf: {}", Message.ShortDebugString());
    return llvm::None;
  }
  clangd::Symbol Result;
  auto ID = SymbolID::fromStr(Message.id());
  if (!ID) {
    elog("Cannot convert parse SymbolID {} from Protobuf: {}", ID.takeError(),
         Message.ShortDebugString());
    return llvm::None;
  }
  Result.ID = *ID;
  Result.SymInfo = fromProtobuf(Message.info());
  Result.Name = Message.name();
  Result.Scope = Message.scope();
  Result.Definition = fromProtobuf(Message.definition(), Strings);
  Result.CanonicalDeclaration =
      fromProtobuf(Message.canonical_declarattion(), Strings);
  Result.References = Message.references();
  Result.Origin = static_cast<clangd::SymbolOrigin>(Message.origin());
  Result.Signature = Message.signature();
  Result.TemplateSpecializationArgs = Message.template_specialization_args();
  Result.CompletionSnippetSuffix = Message.completion_snippet_suffix();
  Result.Documentation = Message.documentation();
  Result.ReturnType = Message.return_type();
  Result.Type = Message.type();
  for (const auto &Header : Message.headers()) {
    Result.IncludeHeaders.push_back(fromProtobuf(Header));
  }
  Result.Flags = static_cast<clangd::Symbol::SymbolFlag>(Message.flags());
  return Result;
}

llvm::Optional<clangd::Ref> fromProtobuf(const Ref &Message,
                                         llvm::UniqueStringSaver *Strings) {
  if (!Message.has_location()) {
    elog("Cannot convert Ref from Protobuf: {}", Message.ShortDebugString());
    return llvm::None;
  }
  clangd::Ref Result;
  Result.Location = fromProtobuf(Message.location(), Strings);
  Result.Kind = static_cast<clangd::RefKind>(Message.kind());
  return Result;
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
  RPCRequest.set_restricted_for_code_completion(From.RestrictForCodeCompletion);
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
  Result.set_id(From.ID.str());
  *Result.mutable_info() = toProtobuf(From.SymInfo);
  Result.set_name(From.Name.str());
  *Result.mutable_definition() = toProtobuf(From.Definition);
  Result.set_scope(From.Scope.str());
  *Result.mutable_canonical_declarattion() =
      toProtobuf(From.CanonicalDeclaration);
  Result.set_references(From.References);
  Result.set_origin(static_cast<uint32_t>(From.Origin));
  Result.set_signature(From.Signature.str());
  Result.set_template_specialization_args(
      From.TemplateSpecializationArgs.str());
  Result.set_completion_snippet_suffix(From.CompletionSnippetSuffix.str());
  Result.set_documentation(From.Documentation.str());
  Result.set_return_type(From.ReturnType.str());
  Result.set_type(From.Type.str());
  for (const auto &Header : From.IncludeHeaders) {
    auto *NextHeader = Result.add_headers();
    *NextHeader = toProtobuf(Header);
  }
  Result.set_flags(static_cast<uint32_t>(From.Flags));
  return Result;
}

Ref toProtobuf(const clangd::Ref &From) {
  Ref Result;
  Result.set_kind(static_cast<uint32_t>(From.Kind));
  *Result.mutable_location() = toProtobuf(From.Location);
  return Result;
}

} // namespace remote
} // namespace clangd
} // namespace clang
