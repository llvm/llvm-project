//===--- Marshalling.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Marshalling.h"
#include "Headers.h"
#include "Index.pb.h"
#include "Protocol.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "index/SymbolLocation.h"
#include "index/SymbolOrigin.h"
#include "support/Logger.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
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

llvm::Optional<clangd::SymbolLocation>
fromProtobuf(const SymbolLocation &Message, llvm::UniqueStringSaver *Strings,
             llvm::StringRef IndexRoot) {
  clangd::SymbolLocation Location;
  auto URIString = relativePathToURI(Message.file_path(), IndexRoot);
  if (!URIString)
    return llvm::None;
  Location.FileURI = Strings->save(*URIString).begin();
  Location.Start = fromProtobuf(Message.start());
  Location.End = fromProtobuf(Message.end());
  return Location;
}

llvm::Optional<SymbolLocation>
toProtobuf(const clangd::SymbolLocation &Location, llvm::StringRef IndexRoot) {
  remote::SymbolLocation Result;
  auto RelativePath = uriToRelativePath(Location.FileURI, IndexRoot);
  if (!RelativePath)
    return llvm::None;
  *Result.mutable_file_path() = *RelativePath;
  *Result.mutable_start() = toProtobuf(Location.Start);
  *Result.mutable_end() = toProtobuf(Location.End);
  return Result;
}

llvm::Optional<HeaderWithReferences>
toProtobuf(const clangd::Symbol::IncludeHeaderWithReferences &IncludeHeader,
           llvm::StringRef IndexRoot) {
  HeaderWithReferences Result;
  Result.set_references(IncludeHeader.References);
  const std::string Header = IncludeHeader.IncludeHeader.str();
  if (isLiteralInclude(Header)) {
    Result.set_header(Header);
    return Result;
  }
  auto RelativePath = uriToRelativePath(Header, IndexRoot);
  if (!RelativePath)
    return llvm::None;
  Result.set_header(*RelativePath);
  return Result;
}

llvm::Optional<clangd::Symbol::IncludeHeaderWithReferences>
fromProtobuf(const HeaderWithReferences &Message,
             llvm::UniqueStringSaver *Strings, llvm::StringRef IndexRoot) {
  std::string Header = Message.header();
  if (Header.front() != '<' && Header.front() != '"') {
    auto URIString = relativePathToURI(Header, IndexRoot);
    if (!URIString)
      return llvm::None;
    Header = *URIString;
  }
  return clangd::Symbol::IncludeHeaderWithReferences{Strings->save(Header),
                                                     Message.references()};
}

} // namespace

clangd::FuzzyFindRequest fromProtobuf(const FuzzyFindRequest *Request,
                                      llvm::StringRef IndexRoot) {
  clangd::FuzzyFindRequest Result;
  Result.Query = Request->query();
  for (const auto &Scope : Request->scopes())
    Result.Scopes.push_back(Scope);
  Result.AnyScope = Request->any_scope();
  if (Request->limit())
    Result.Limit = Request->limit();
  Result.RestrictForCodeCompletion = Request->restricted_for_code_completion();
  for (const auto &Path : Request->proximity_paths()) {
    llvm::SmallString<256> LocalPath = llvm::StringRef(IndexRoot);
    llvm::sys::path::append(LocalPath, Path);
    Result.ProximityPaths.push_back(std::string(LocalPath));
  }
  for (const auto &Type : Request->preferred_types())
    Result.ProximityPaths.push_back(Type);
  return Result;
}

llvm::Optional<clangd::Symbol> fromProtobuf(const Symbol &Message,
                                            llvm::UniqueStringSaver *Strings,
                                            llvm::StringRef IndexRoot) {
  if (!Message.has_info() || !Message.has_definition() ||
      !Message.has_canonical_declaration()) {
    elog("Cannot convert Symbol from Protobuf: {0}",
         Message.ShortDebugString());
    return llvm::None;
  }
  clangd::Symbol Result;
  auto ID = SymbolID::fromStr(Message.id());
  if (!ID) {
    elog("Cannot parse SymbolID {0} given Protobuf: {1}", ID.takeError(),
         Message.ShortDebugString());
    return llvm::None;
  }
  Result.ID = *ID;
  Result.SymInfo = fromProtobuf(Message.info());
  Result.Name = Message.name();
  Result.Scope = Message.scope();
  auto Definition = fromProtobuf(Message.definition(), Strings, IndexRoot);
  if (!Definition)
    return llvm::None;
  Result.Definition = *Definition;
  auto Declaration =
      fromProtobuf(Message.canonical_declaration(), Strings, IndexRoot);
  if (!Declaration)
    return llvm::None;
  Result.CanonicalDeclaration = *Declaration;
  Result.References = Message.references();
  Result.Origin = static_cast<clangd::SymbolOrigin>(Message.origin());
  Result.Signature = Message.signature();
  Result.TemplateSpecializationArgs = Message.template_specialization_args();
  Result.CompletionSnippetSuffix = Message.completion_snippet_suffix();
  Result.Documentation = Message.documentation();
  Result.ReturnType = Message.return_type();
  Result.Type = Message.type();
  for (const auto &Header : Message.headers()) {
    auto SerializedHeader = fromProtobuf(Header, Strings, IndexRoot);
    if (SerializedHeader)
      Result.IncludeHeaders.push_back(*SerializedHeader);
  }
  Result.Flags = static_cast<clangd::Symbol::SymbolFlag>(Message.flags());
  return Result;
}

llvm::Optional<clangd::Ref> fromProtobuf(const Ref &Message,
                                         llvm::UniqueStringSaver *Strings,
                                         llvm::StringRef IndexRoot) {
  if (!Message.has_location()) {
    elog("Cannot convert Ref from Protobuf: {}", Message.ShortDebugString());
    return llvm::None;
  }
  clangd::Ref Result;
  auto Location = fromProtobuf(Message.location(), Strings, IndexRoot);
  if (!Location)
    return llvm::None;
  Result.Location = *Location;
  Result.Kind = static_cast<clangd::RefKind>(Message.kind());
  return Result;
}

LookupRequest toProtobuf(const clangd::LookupRequest &From) {
  LookupRequest RPCRequest;
  for (const auto &SymbolID : From.IDs)
    RPCRequest.add_ids(SymbolID.str());
  return RPCRequest;
}

FuzzyFindRequest toProtobuf(const clangd::FuzzyFindRequest &From,
                            llvm::StringRef IndexRoot) {
  FuzzyFindRequest RPCRequest;
  RPCRequest.set_query(From.Query);
  for (const auto &Scope : From.Scopes)
    RPCRequest.add_scopes(Scope);
  RPCRequest.set_any_scope(From.AnyScope);
  if (From.Limit)
    RPCRequest.set_limit(*From.Limit);
  RPCRequest.set_restricted_for_code_completion(From.RestrictForCodeCompletion);
  for (const auto &Path : From.ProximityPaths) {
    llvm::SmallString<256> RelativePath = llvm::StringRef(Path);
    if (llvm::sys::path::replace_path_prefix(RelativePath, IndexRoot, ""))
      RPCRequest.add_proximity_paths(llvm::sys::path::convert_to_slash(
          RelativePath, llvm::sys::path::Style::posix));
  }
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

Symbol toProtobuf(const clangd::Symbol &From, llvm::StringRef IndexRoot) {
  Symbol Result;
  Result.set_id(From.ID.str());
  *Result.mutable_info() = toProtobuf(From.SymInfo);
  Result.set_name(From.Name.str());
  auto Definition = toProtobuf(From.Definition, IndexRoot);
  if (Definition)
    *Result.mutable_definition() = *Definition;
  Result.set_scope(From.Scope.str());
  auto Declaration = toProtobuf(From.CanonicalDeclaration, IndexRoot);
  if (Declaration)
    *Result.mutable_canonical_declaration() = *Declaration;
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
    auto Serialized = toProtobuf(Header, IndexRoot);
    if (!Serialized)
      continue;
    auto *NextHeader = Result.add_headers();
    *NextHeader = *Serialized;
  }
  Result.set_flags(static_cast<uint32_t>(From.Flags));
  return Result;
}

// FIXME(kirillbobyrev): A reference without location is invalid.
// llvm::Optional<Ref> here and on the server side?
Ref toProtobuf(const clangd::Ref &From, llvm::StringRef IndexRoot) {
  Ref Result;
  Result.set_kind(static_cast<uint32_t>(From.Kind));
  auto Location = toProtobuf(From.Location, IndexRoot);
  if (Location)
    *Result.mutable_location() = *Location;
  return Result;
}

llvm::Optional<std::string> relativePathToURI(llvm::StringRef RelativePath,
                                              llvm::StringRef IndexRoot) {
  assert(RelativePath == llvm::sys::path::convert_to_slash(
                             RelativePath, llvm::sys::path::Style::posix));
  assert(IndexRoot == llvm::sys::path::convert_to_slash(IndexRoot));
  assert(IndexRoot.endswith(llvm::sys::path::get_separator()));
  if (RelativePath.empty())
    return std::string();
  if (llvm::sys::path::is_absolute(RelativePath)) {
    elog("Remote index client got absolute path from server: {0}",
         RelativePath);
    return llvm::None;
  }
  if (llvm::sys::path::is_relative(IndexRoot)) {
    elog("Remote index client got a relative path as index root: {0}",
         IndexRoot);
    return llvm::None;
  }
  llvm::SmallString<256> FullPath = IndexRoot;
  llvm::sys::path::append(FullPath, RelativePath);
  auto Result = URI::createFile(FullPath);
  return Result.toString();
}

llvm::Optional<std::string> uriToRelativePath(llvm::StringRef URI,
                                              llvm::StringRef IndexRoot) {
  assert(IndexRoot.endswith(llvm::sys::path::get_separator()));
  assert(IndexRoot == llvm::sys::path::convert_to_slash(IndexRoot));
  assert(!IndexRoot.empty());
  if (llvm::sys::path::is_relative(IndexRoot)) {
    elog("Index root {0} is not absolute path", IndexRoot);
    return llvm::None;
  }
  auto ParsedURI = URI::parse(URI);
  if (!ParsedURI) {
    elog("Remote index got bad URI from client {0}: {1}", URI,
         ParsedURI.takeError());
    return llvm::None;
  }
  if (ParsedURI->scheme() != "file") {
    elog("Remote index got URI with scheme other than \"file\" {0}: {1}", URI,
         ParsedURI->scheme());
    return llvm::None;
  }
  llvm::SmallString<256> Result = ParsedURI->body();
  if (!llvm::sys::path::replace_path_prefix(Result, IndexRoot, "")) {
    elog("Can not get relative path from the URI {0} given the index root {1}",
         URI, IndexRoot);
    return llvm::None;
  }
  // Make sure the result has UNIX slashes.
  return llvm::sys::path::convert_to_slash(Result,
                                           llvm::sys::path::Style::posix);
}

} // namespace remote
} // namespace clangd
} // namespace clang
