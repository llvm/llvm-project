//===--- ConfigYAML.cpp - Loading configuration fragments from YAML files -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConfigFragment.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include <system_error>

namespace clang {
namespace clangd {
namespace config {
namespace {
using llvm::yaml::BlockScalarNode;
using llvm::yaml::MappingNode;
using llvm::yaml::Node;
using llvm::yaml::ScalarNode;
using llvm::yaml::SequenceNode;

class Parser {
  llvm::SourceMgr &SM;
  bool HadError = false;

public:
  Parser(llvm::SourceMgr &SM) : SM(SM) {}

  // Tries to parse N into F, returning false if it failed and we couldn't
  // meaningfully recover (YAML syntax error, or hard semantic error).
  bool parse(Fragment &F, Node &N) {
    DictParser Dict("Config", this);
    Dict.handle("If", [&](Node &N) { parse(F.If, N); });
    Dict.handle("CompileFlags", [&](Node &N) { parse(F.CompileFlags, N); });
    Dict.handle("Index", [&](Node &N) { parse(F.Index, N); });
    Dict.parse(N);
    return !(N.failed() || HadError);
  }

private:
  void parse(Fragment::IfBlock &F, Node &N) {
    DictParser Dict("If", this);
    Dict.unrecognized(
        [&](llvm::StringRef) { F.HasUnrecognizedCondition = true; });
    Dict.handle("PathMatch", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.PathMatch = std::move(*Values);
    });
    Dict.handle("PathExclude", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.PathExclude = std::move(*Values);
    });
    Dict.parse(N);
  }

  void parse(Fragment::CompileFlagsBlock &F, Node &N) {
    DictParser Dict("CompileFlags", this);
    Dict.handle("Add", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.Add = std::move(*Values);
    });
    Dict.handle("Remove", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.Remove = std::move(*Values);
    });
    Dict.parse(N);
  }

  void parse(Fragment::IndexBlock &F, Node &N) {
    DictParser Dict("Index", this);
    Dict.handle("Background",
                [&](Node &N) { F.Background = scalarValue(N, "Background"); });
    Dict.parse(N);
  }

  // Helper for parsing mapping nodes (dictionaries).
  // We don't use YamlIO as we want to control over unknown keys.
  class DictParser {
    llvm::StringRef Description;
    std::vector<std::pair<llvm::StringRef, std::function<void(Node &)>>> Keys;
    std::function<void(llvm::StringRef)> Unknown;
    Parser *Outer;

  public:
    DictParser(llvm::StringRef Description, Parser *Outer)
        : Description(Description), Outer(Outer) {}

    // Parse is called when Key is encountered, and passed the associated value.
    // It should emit diagnostics if the value is invalid (e.g. wrong type).
    // If Key is seen twice, Parse runs only once and an error is reported.
    void handle(llvm::StringLiteral Key, std::function<void(Node &)> Parse) {
      for (const auto &Entry : Keys) {
        (void) Entry;
        assert(Entry.first != Key && "duplicate key handler");
      }
      Keys.emplace_back(Key, std::move(Parse));
    }

    // Fallback is called when a Key is not matched by any handle().
    // A warning is also automatically emitted.
    void unrecognized(std::function<void(llvm::StringRef)> Fallback) {
      Unknown = std::move(Fallback);
    }

    // Process a mapping node and call handlers for each key/value pair.
    void parse(Node &N) const {
      if (N.getType() != Node::NK_Mapping) {
        Outer->error(Description + " should be a dictionary", N);
        return;
      }
      llvm::SmallSet<std::string, 8> Seen;
      // We *must* consume all items, even on error, or the parser will assert.
      for (auto &KV : llvm::cast<MappingNode>(N)) {
        auto *K = KV.getKey();
        if (!K) // YAMLParser emitted an error.
          continue;
        auto Key = Outer->scalarValue(*K, "Dictionary key");
        if (!Key)
          continue;
        if (!Seen.insert(**Key).second) {
          Outer->warning("Duplicate key " + **Key + " is ignored", *K);
          continue;
        }
        auto *Value = KV.getValue();
        if (!Value) // YAMLParser emitted an error.
          continue;
        bool Matched = false;
        for (const auto &Handler : Keys) {
          if (Handler.first == **Key) {
            Matched = true;
            Handler.second(*Value);
            break;
          }
        }
        if (!Matched) {
          Outer->warning("Unknown " + Description + " key " + **Key, *K);
          if (Unknown)
            Unknown(**Key);
        }
      }
    }
  };

  // Try to parse a single scalar value from the node, warn on failure.
  llvm::Optional<Located<std::string>> scalarValue(Node &N,
                                                   llvm::StringRef Desc) {
    llvm::SmallString<256> Buf;
    if (auto *S = llvm::dyn_cast<ScalarNode>(&N))
      return Located<std::string>(S->getValue(Buf).str(), N.getSourceRange());
    if (auto *BS = llvm::dyn_cast<BlockScalarNode>(&N))
      return Located<std::string>(BS->getValue().str(), N.getSourceRange());
    warning(Desc + " should be scalar", N);
    return llvm::None;
  }

  // Try to parse a list of single scalar values, or just a single value.
  llvm::Optional<std::vector<Located<std::string>>> scalarValues(Node &N) {
    std::vector<Located<std::string>> Result;
    if (auto *S = llvm::dyn_cast<ScalarNode>(&N)) {
      llvm::SmallString<256> Buf;
      Result.emplace_back(S->getValue(Buf).str(), N.getSourceRange());
    } else if (auto *S = llvm::dyn_cast<BlockScalarNode>(&N)) {
      Result.emplace_back(S->getValue().str(), N.getSourceRange());
    } else if (auto *S = llvm::dyn_cast<SequenceNode>(&N)) {
      // We *must* consume all items, even on error, or the parser will assert.
      for (auto &Child : *S) {
        if (auto Value = scalarValue(Child, "List item"))
          Result.push_back(std::move(*Value));
      }
    } else {
      warning("Expected scalar or list of scalars", N);
      return llvm::None;
    }
    return Result;
  }

  // Report a "hard" error, reflecting a config file that can never be valid.
  void error(const llvm::Twine &Msg, const Node &N) {
    HadError = true;
    SM.PrintMessage(N.getSourceRange().Start, llvm::SourceMgr::DK_Error, Msg,
                    N.getSourceRange());
  }

  // Report a "soft" error that could be caused by e.g. version skew.
  void warning(const llvm::Twine &Msg, const Node &N) {
    SM.PrintMessage(N.getSourceRange().Start, llvm::SourceMgr::DK_Warning, Msg,
                    N.getSourceRange());
  }
};

} // namespace

std::vector<Fragment> Fragment::parseYAML(llvm::StringRef YAML,
                                          llvm::StringRef BufferName,
                                          DiagnosticCallback Diags) {
  // The YAML document may contain multiple conditional fragments.
  // The SourceManager is shared for all of them.
  auto SM = std::make_shared<llvm::SourceMgr>();
  auto Buf = llvm::MemoryBuffer::getMemBufferCopy(YAML, BufferName);
  // Adapt DiagnosticCallback to function-pointer interface.
  // Callback receives both errors we emit and those from the YAML parser.
  SM->setDiagHandler(
      [](const llvm::SMDiagnostic &Diag, void *Ctx) {
        (*reinterpret_cast<DiagnosticCallback *>(Ctx))(Diag);
      },
      &Diags);
  std::vector<Fragment> Result;
  for (auto &Doc : llvm::yaml::Stream(*Buf, *SM)) {
    if (Node *N = Doc.getRoot()) {
      Fragment Fragment;
      Fragment.Source.Manager = SM;
      Fragment.Source.Location = N->getSourceRange().Start;
      if (Parser(*SM).parse(Fragment, *N))
        Result.push_back(std::move(Fragment));
    }
  }
  // Hack: stash the buffer in the SourceMgr to keep it alive.
  // SM has two entries: "main" non-owning buffer, and ignored owning buffer.
  SM->AddNewSourceBuffer(std::move(Buf), llvm::SMLoc());
  return Result;
}

} // namespace config
} // namespace clangd
} // namespace clang
