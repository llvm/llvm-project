//===--- PathMapping.cpp - apply path mappings to LSP messages -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PathMapping.h"
#include "Transport.h"
#include "URI.h"
#include "support/Logger.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cctype>
#include <optional>
#include <tuple>

namespace clang {
namespace clangd {
std::optional<std::string> doPathMapping(llvm::StringRef S,
                                         PathMapping::Direction Dir,
                                         const PathMappings &Mappings) {
  // Return early to optimize for the common case, wherein S is not a file URI
  if (!S.starts_with("file://"))
    return std::nullopt;
  auto Uri = URI::parse(S);
  if (!Uri) {
    llvm::consumeError(Uri.takeError());
    return std::nullopt;
  }

  std::string BodyStr = (*Uri).body().str();
  std::replace(BodyStr.begin(), BodyStr.end(), '\\', '/');

  bool DidPreferEmbeddedDrive = false;
  if (BodyStr.rfind("/mnt/", 0) == 0 && BodyStr.size() > 6) {
    for (size_t i = 0; i + 2 < BodyStr.size(); ++i) {
      if (std::isalpha((unsigned char)BodyStr[i]) && BodyStr[i + 1] == ':' &&
          BodyStr[i + 2] == '/') {
        BodyStr = std::string("/") + BodyStr.substr(i);
        DidPreferEmbeddedDrive = true;
        break;
      }
    }
  }

  if (Dir == PathMapping::Direction::ClientToServer && DidPreferEmbeddedDrive) {
    if (BodyStr.size() >= 3 && BodyStr[0] == '/' &&
        std::isalpha((unsigned char)BodyStr[1]) && BodyStr[2] == ':') {
      return URI((*Uri).scheme(), (*Uri).authority(), BodyStr).toString();
    }
  }

  llvm::StringRef BodyRef(BodyStr);
  for (const auto &Mapping : Mappings) {
    const std::string &From = Dir == PathMapping::Direction::ClientToServer
                                  ? Mapping.ClientPath
                                  : Mapping.ServerPath;
    const std::string &To = Dir == PathMapping::Direction::ClientToServer
                                ? Mapping.ServerPath
                                : Mapping.ClientPath;
    llvm::StringRef Working = BodyRef;
    if (Working.consume_front(From)) {
      if (From.empty() || From.back() == '/' || Working.empty() ||
          Working.front() == '/') {
        llvm::StringRef Adjusted = Working;

        char MappingDrive = 0;
        if (To.size() >= 3 && To[0] == '/' &&
            std::isalpha((unsigned char)To[1]) && To[2] == ':')
          MappingDrive = To[1];
        if (MappingDrive) {
          for (size_t i = 0; i + 2 < (size_t)Working.size(); ++i) {
            char c = Working[i];
            if (std::isalpha((unsigned char)c) && Working[i + 1] == ':' &&
                Working[i + 2] == '/') {
              if (std::tolower((unsigned char)c) ==
                  std::tolower((unsigned char)MappingDrive)) {
                Adjusted = Working.substr(i + 3);
                break;
              }
            }
          }
        }
        std::string MappedBody = (To + Adjusted).str();
        return URI((*Uri).scheme(), (*Uri).authority(), MappedBody).toString();
      }
    }
  }

  if (Dir == PathMapping::Direction::ServerToClient) {
    for (const auto &Mapping : Mappings) {
      const std::string &From = Mapping.ServerPath;
      const std::string &To = Mapping.ClientPath;
      if (From.empty())
        continue;
      llvm::StringRef W = BodyRef;
      if (W.starts_with(From)) {
        llvm::StringRef Rest = W.substr(From.size());
        if (Rest.empty() || Rest.front() == '/') {
          std::string MappedBody = (To + Rest).str();
          return URI((*Uri).scheme(), (*Uri).authority(), MappedBody)
              .toString();
        }
      }
    }
  }

  return std::nullopt;
}

void applyPathMappings(llvm::json::Value &V, PathMapping::Direction Dir,
                       const PathMappings &Mappings) {
  using Kind = llvm::json::Value::Kind;
  Kind K = V.kind();
  if (K == Kind::Object) {
    llvm::json::Object *Obj = V.getAsObject();
    llvm::json::Object MappedObj;
    // 1. Map all the Keys
    for (auto &KV : *Obj) {
      if (std::optional<std::string> MappedKey =
              doPathMapping(KV.first.str(), Dir, Mappings)) {
        MappedObj.try_emplace(std::move(*MappedKey), std::move(KV.second));
      } else {
        MappedObj.try_emplace(std::move(KV.first), std::move(KV.second));
      }
    }
    *Obj = std::move(MappedObj);
    // 2. Map all the values
    for (auto &KV : *Obj)
      applyPathMappings(KV.second, Dir, Mappings);
  } else if (K == Kind::Array) {
    for (llvm::json::Value &Val : *V.getAsArray())
      applyPathMappings(Val, Dir, Mappings);
  } else if (K == Kind::String) {
    if (std::optional<std::string> Mapped =
            doPathMapping(*V.getAsString(), Dir, Mappings))
      V = std::move(*Mapped);
  }
}

namespace {

class PathMappingMessageHandler : public Transport::MessageHandler {
public:
  PathMappingMessageHandler(MessageHandler &Handler,
                            const PathMappings &Mappings)
      : WrappedHandler(Handler), Mappings(Mappings) {}

  bool onNotify(llvm::StringRef Method, llvm::json::Value Params) override {
    applyPathMappings(Params, PathMapping::Direction::ClientToServer, Mappings);
    return WrappedHandler.onNotify(Method, std::move(Params));
  }

  bool onCall(llvm::StringRef Method, llvm::json::Value Params,
              llvm::json::Value ID) override {
    applyPathMappings(Params, PathMapping::Direction::ClientToServer, Mappings);
    return WrappedHandler.onCall(Method, std::move(Params), std::move(ID));
  }

  bool onReply(llvm::json::Value ID,
               llvm::Expected<llvm::json::Value> Result) override {
    if (Result)
      applyPathMappings(*Result, PathMapping::Direction::ClientToServer,
                        Mappings);
    return WrappedHandler.onReply(std::move(ID), std::move(Result));
  }

private:
  Transport::MessageHandler &WrappedHandler;
  const PathMappings &Mappings;
};

// Apply path mappings to all LSP messages by intercepting all params/results
// and then delegating to the normal transport
class PathMappingTransport : public Transport {
public:
  PathMappingTransport(std::unique_ptr<Transport> Transp, PathMappings Mappings)
      : WrappedTransport(std::move(Transp)), Mappings(std::move(Mappings)) {}

  void notify(llvm::StringRef Method, llvm::json::Value Params) override {
    applyPathMappings(Params, PathMapping::Direction::ServerToClient, Mappings);
    WrappedTransport->notify(Method, std::move(Params));
  }

  void call(llvm::StringRef Method, llvm::json::Value Params,
            llvm::json::Value ID) override {
    applyPathMappings(Params, PathMapping::Direction::ServerToClient, Mappings);
    WrappedTransport->call(Method, std::move(Params), std::move(ID));
  }

  void reply(llvm::json::Value ID,
             llvm::Expected<llvm::json::Value> Result) override {
    if (Result)
      applyPathMappings(*Result, PathMapping::Direction::ServerToClient,
                        Mappings);
    WrappedTransport->reply(std::move(ID), std::move(Result));
  }

  llvm::Error loop(MessageHandler &Handler) override {
    PathMappingMessageHandler WrappedHandler(Handler, Mappings);
    return WrappedTransport->loop(WrappedHandler);
  }

private:
  std::unique_ptr<Transport> WrappedTransport;
  PathMappings Mappings;
};

// Converts a unix/windows path to the path portion of a file URI
// e.g. "C:\foo" -> "/C:/foo"
llvm::Expected<std::string> parsePath(llvm::StringRef Path) {
  namespace path = llvm::sys::path;
  if (path::is_absolute(Path, path::Style::posix)) {
    return std::string(Path);
  }
  if (path::is_absolute(Path, path::Style::windows)) {
    std::string Converted = path::convert_to_slash(Path, path::Style::windows);
    if (Converted.front() != '/')
      Converted = "/" + Converted;
    return Converted;
  }
  return error("Path not absolute: {0}", Path);
}

} // namespace

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const PathMapping &M) {
  return OS << M.ClientPath << "=" << M.ServerPath;
}

llvm::Expected<PathMappings>
parsePathMappings(llvm::StringRef RawPathMappings) {
  llvm::StringRef ClientPath, ServerPath, PathPair, Rest = RawPathMappings;
  PathMappings ParsedMappings;
  while (!Rest.empty()) {
    std::tie(PathPair, Rest) = Rest.split(",");
    std::tie(ClientPath, ServerPath) = PathPair.split("=");
    if (ClientPath.empty() || ServerPath.empty())
      return error("Not a valid path mapping pair: {0}", PathPair);
    llvm::Expected<std::string> ParsedClientPath = parsePath(ClientPath);
    if (!ParsedClientPath)
      return ParsedClientPath.takeError();
    llvm::Expected<std::string> ParsedServerPath = parsePath(ServerPath);
    if (!ParsedServerPath)
      return ParsedServerPath.takeError();
    ParsedMappings.push_back(
        {std::move(*ParsedClientPath), std::move(*ParsedServerPath)});
  }
  return ParsedMappings;
}

std::unique_ptr<Transport>
createPathMappingTransport(std::unique_ptr<Transport> Transp,
                           PathMappings Mappings) {
  return std::make_unique<PathMappingTransport>(std::move(Transp), Mappings);
}

} // namespace clangd
} // namespace clang
