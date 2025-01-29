//===--- ConfigYAML.cpp - Loading configuration fragments from YAML files -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ConfigFragment.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include <optional>
#include <string>

namespace clang {
namespace clangd {
namespace config {
namespace {
using llvm::yaml::BlockScalarNode;
using llvm::yaml::MappingNode;
using llvm::yaml::Node;
using llvm::yaml::ScalarNode;
using llvm::yaml::SequenceNode;

std::optional<llvm::StringRef>
bestGuess(llvm::StringRef Search,
          llvm::ArrayRef<llvm::StringRef> AllowedValues) {
  unsigned MaxEdit = (Search.size() + 1) / 3;
  if (!MaxEdit)
    return std::nullopt;
  std::optional<llvm::StringRef> Result;
  for (const auto &AllowedValue : AllowedValues) {
    unsigned EditDistance = Search.edit_distance(AllowedValue, true, MaxEdit);
    // We can't do better than an edit distance of 1, so just return this and
    // save computing other values.
    if (EditDistance == 1U)
      return AllowedValue;
    if (EditDistance == MaxEdit && !Result) {
      Result = AllowedValue;
    } else if (EditDistance < MaxEdit) {
      Result = AllowedValue;
      MaxEdit = EditDistance;
    }
  }
  return Result;
}

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
    Dict.handle("Style", [&](Node &N) { parse(F.Style, N); });
    Dict.handle("Diagnostics", [&](Node &N) { parse(F.Diagnostics, N); });
    Dict.handle("Completion", [&](Node &N) { parse(F.Completion, N); });
    Dict.handle("Hover", [&](Node &N) { parse(F.Hover, N); });
    Dict.handle("InlayHints", [&](Node &N) { parse(F.InlayHints, N); });
    Dict.handle("SemanticTokens", [&](Node &N) { parse(F.SemanticTokens, N); });
    Dict.parse(N);
    return !(N.failed() || HadError);
  }

private:
  void parse(Fragment::IfBlock &F, Node &N) {
    DictParser Dict("If", this);
    Dict.unrecognized([&](Located<std::string>, Node &) {
      F.HasUnrecognizedCondition = true;
      return true; // Emit a warning for the unrecognized key.
    });
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
    Dict.handle("Compiler", [&](Node &N) {
      if (auto Value = scalarValue(N, "Compiler"))
        F.Compiler = std::move(*Value);
    });
    Dict.handle("Add", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.Add = std::move(*Values);
    });
    Dict.handle("Remove", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.Remove = std::move(*Values);
    });
    Dict.handle("CompilationDatabase", [&](Node &N) {
      F.CompilationDatabase = scalarValue(N, "CompilationDatabase");
    });
    Dict.parse(N);
  }

  void parse(Fragment::StyleBlock &F, Node &N) {
    DictParser Dict("Style", this);
    Dict.handle("FullyQualifiedNamespaces", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.FullyQualifiedNamespaces = std::move(*Values);
    });
    Dict.handle("QuotedHeaders", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.QuotedHeaders = std::move(*Values);
    });
    Dict.handle("AngledHeaders", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.AngledHeaders = std::move(*Values);
    });
    Dict.parse(N);
  }

  void parse(Fragment::DiagnosticsBlock &F, Node &N) {
    DictParser Dict("Diagnostics", this);
    Dict.handle("Suppress", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.Suppress = std::move(*Values);
    });
    Dict.handle("UnusedIncludes", [&](Node &N) {
      F.UnusedIncludes = scalarValue(N, "UnusedIncludes");
    });
    Dict.handle("MissingIncludes", [&](Node &N) {
      F.MissingIncludes = scalarValue(N, "MissingIncludes");
    });
    Dict.handle("Includes", [&](Node &N) { parse(F.Includes, N); });
    Dict.handle("ClangTidy", [&](Node &N) { parse(F.ClangTidy, N); });
    Dict.parse(N);
  }

  void parse(Fragment::DiagnosticsBlock::ClangTidyBlock &F, Node &N) {
    DictParser Dict("ClangTidy", this);
    Dict.handle("Add", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.Add = std::move(*Values);
    });
    Dict.handle("Remove", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.Remove = std::move(*Values);
    });
    Dict.handle("CheckOptions", [&](Node &N) {
      DictParser CheckOptDict("CheckOptions", this);
      CheckOptDict.unrecognized([&](Located<std::string> &&Key, Node &Val) {
        if (auto Value = scalarValue(Val, *Key))
          F.CheckOptions.emplace_back(std::move(Key), std::move(*Value));
        return false; // Don't emit a warning
      });
      CheckOptDict.parse(N);
    });
    Dict.handle("FastCheckFilter", [&](Node &N) {
      if (auto FastCheckFilter = scalarValue(N, "FastCheckFilter"))
        F.FastCheckFilter = *FastCheckFilter;
    });
    Dict.parse(N);
  }

  void parse(Fragment::DiagnosticsBlock::IncludesBlock &F, Node &N) {
    DictParser Dict("Includes", this);
    Dict.handle("IgnoreHeader", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.IgnoreHeader = std::move(*Values);
    });
    Dict.handle("AnalyzeAngledIncludes", [&](Node &N) {
      if (auto Value = boolValue(N, "AnalyzeAngledIncludes"))
        F.AnalyzeAngledIncludes = *Value;
    });
    Dict.parse(N);
  }

  void parse(Fragment::IndexBlock &F, Node &N) {
    DictParser Dict("Index", this);
    Dict.handle("Background",
                [&](Node &N) { F.Background = scalarValue(N, "Background"); });
    Dict.handle("External", [&](Node &N) {
      Fragment::IndexBlock::ExternalBlock External;
      // External block can either be a mapping or a scalar value. Dispatch
      // accordingly.
      if (N.getType() == Node::NK_Mapping) {
        parse(External, N);
      } else if (N.getType() == Node::NK_Scalar ||
                 N.getType() == Node::NK_BlockScalar) {
        parse(External, *scalarValue(N, "External"));
      } else {
        error("External must be either a scalar or a mapping.", N);
        return;
      }
      F.External.emplace(std::move(External));
      F.External->Range = N.getSourceRange();
    });
    Dict.handle("StandardLibrary", [&](Node &N) {
      if (auto StandardLibrary = boolValue(N, "StandardLibrary"))
        F.StandardLibrary = *StandardLibrary;
    });
    Dict.parse(N);
  }

  void parse(Fragment::IndexBlock::ExternalBlock &F,
             Located<std::string> ExternalVal) {
    if (!llvm::StringRef(*ExternalVal).equals_insensitive("none")) {
      error("Only scalar value supported for External is 'None'",
            ExternalVal.Range);
      return;
    }
    F.IsNone = true;
    F.IsNone.Range = ExternalVal.Range;
  }

  void parse(Fragment::IndexBlock::ExternalBlock &F, Node &N) {
    DictParser Dict("External", this);
    Dict.handle("File", [&](Node &N) { F.File = scalarValue(N, "File"); });
    Dict.handle("Server",
                [&](Node &N) { F.Server = scalarValue(N, "Server"); });
    Dict.handle("MountPoint",
                [&](Node &N) { F.MountPoint = scalarValue(N, "MountPoint"); });
    Dict.parse(N);
  }

  void parse(Fragment::CompletionBlock &F, Node &N) {
    DictParser Dict("Completion", this);
    Dict.handle("AllScopes", [&](Node &N) {
      if (auto AllScopes = boolValue(N, "AllScopes"))
        F.AllScopes = *AllScopes;
    });
    Dict.handle("ArgumentLists", [&](Node &N) {
      if (auto ArgumentLists = scalarValue(N, "ArgumentLists"))
        F.ArgumentLists = *ArgumentLists;
    });
    Dict.parse(N);
  }

  void parse(Fragment::HoverBlock &F, Node &N) {
    DictParser Dict("Hover", this);
    Dict.handle("ShowAKA", [&](Node &N) {
      if (auto ShowAKA = boolValue(N, "ShowAKA"))
        F.ShowAKA = *ShowAKA;
    });
    Dict.parse(N);
  }

  void parse(Fragment::InlayHintsBlock &F, Node &N) {
    DictParser Dict("InlayHints", this);
    Dict.handle("Enabled", [&](Node &N) {
      if (auto Value = boolValue(N, "Enabled"))
        F.Enabled = *Value;
    });
    Dict.handle("ParameterNames", [&](Node &N) {
      if (auto Value = boolValue(N, "ParameterNames"))
        F.ParameterNames = *Value;
    });
    Dict.handle("DeducedTypes", [&](Node &N) {
      if (auto Value = boolValue(N, "DeducedTypes"))
        F.DeducedTypes = *Value;
    });
    Dict.handle("Designators", [&](Node &N) {
      if (auto Value = boolValue(N, "Designators"))
        F.Designators = *Value;
    });
    Dict.handle("BlockEnd", [&](Node &N) {
      if (auto Value = boolValue(N, "BlockEnd"))
        F.BlockEnd = *Value;
    });
    Dict.handle("DefaultArguments", [&](Node &N) {
      if (auto Value = boolValue(N, "DefaultArguments"))
        F.DefaultArguments = *Value;
    });
    Dict.handle("TypeNameLimit", [&](Node &N) {
      if (auto Value = uint32Value(N, "TypeNameLimit"))
        F.TypeNameLimit = *Value;
    });
    Dict.parse(N);
  }

  void parse(Fragment::SemanticTokensBlock &F, Node &N) {
    DictParser Dict("SemanticTokens", this);
    Dict.handle("DisabledKinds", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.DisabledKinds = std::move(*Values);
    });
    Dict.handle("DisabledModifiers", [&](Node &N) {
      if (auto Values = scalarValues(N))
        F.DisabledModifiers = std::move(*Values);
    });
    Dict.parse(N);
  }

  // Helper for parsing mapping nodes (dictionaries).
  // We don't use YamlIO as we want to control over unknown keys.
  class DictParser {
    llvm::StringRef Description;
    std::vector<std::pair<llvm::StringRef, std::function<void(Node &)>>> Keys;
    std::function<bool(Located<std::string>, Node &)> UnknownHandler;
    Parser *Outer;

  public:
    DictParser(llvm::StringRef Description, Parser *Outer)
        : Description(Description), Outer(Outer) {}

    // Parse is called when Key is encountered, and passed the associated value.
    // It should emit diagnostics if the value is invalid (e.g. wrong type).
    // If Key is seen twice, Parse runs only once and an error is reported.
    void handle(llvm::StringLiteral Key, std::function<void(Node &)> Parse) {
      for (const auto &Entry : Keys) {
        (void)Entry;
        assert(Entry.first != Key && "duplicate key handler");
      }
      Keys.emplace_back(Key, std::move(Parse));
    }

    // Handler is called when a Key is not matched by any handle().
    // If this is unset or the Handler returns true, a warning is emitted for
    // the unknown key.
    void
    unrecognized(std::function<bool(Located<std::string>, Node &)> Handler) {
      UnknownHandler = std::move(Handler);
    }

    // Process a mapping node and call handlers for each key/value pair.
    void parse(Node &N) const {
      if (N.getType() != Node::NK_Mapping) {
        Outer->error(Description + " should be a dictionary", N);
        return;
      }
      llvm::SmallSet<std::string, 8> Seen;
      llvm::SmallVector<Located<std::string>, 0> UnknownKeys;
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
          if (auto *Value = KV.getValue())
            Value->skip();
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
          bool Warn = !UnknownHandler;
          if (UnknownHandler)
            Warn = UnknownHandler(
                Located<std::string>(**Key, K->getSourceRange()), *Value);
          if (Warn)
            UnknownKeys.push_back(std::move(*Key));
        }
      }
      if (!UnknownKeys.empty())
        warnUnknownKeys(UnknownKeys, Seen);
    }

  private:
    void warnUnknownKeys(llvm::ArrayRef<Located<std::string>> UnknownKeys,
                         const llvm::SmallSet<std::string, 8> &SeenKeys) const {
      llvm::SmallVector<llvm::StringRef> UnseenKeys;
      for (const auto &KeyAndHandler : Keys)
        if (!SeenKeys.count(KeyAndHandler.first.str()))
          UnseenKeys.push_back(KeyAndHandler.first);

      for (const Located<std::string> &UnknownKey : UnknownKeys)
        if (auto BestGuess = bestGuess(*UnknownKey, UnseenKeys))
          Outer->warning("Unknown " + Description + " key '" + *UnknownKey +
                             "'; did you mean '" + *BestGuess + "'?",
                         UnknownKey.Range);
        else
          Outer->warning("Unknown " + Description + " key '" + *UnknownKey +
                             "'",
                         UnknownKey.Range);
    }
  };

  // Try to parse a single scalar value from the node, warn on failure.
  std::optional<Located<std::string>> scalarValue(Node &N,
                                                  llvm::StringRef Desc) {
    llvm::SmallString<256> Buf;
    if (auto *S = llvm::dyn_cast<ScalarNode>(&N))
      return Located<std::string>(S->getValue(Buf).str(), N.getSourceRange());
    if (auto *BS = llvm::dyn_cast<BlockScalarNode>(&N))
      return Located<std::string>(BS->getValue().str(), N.getSourceRange());
    warning(Desc + " should be scalar", N);
    return std::nullopt;
  }

  std::optional<Located<bool>> boolValue(Node &N, llvm::StringRef Desc) {
    if (auto Scalar = scalarValue(N, Desc)) {
      if (auto Bool = llvm::yaml::parseBool(**Scalar))
        return Located<bool>(*Bool, Scalar->Range);
      warning(Desc + " should be a boolean", N);
    }
    return std::nullopt;
  }

  std::optional<Located<uint32_t>> uint32Value(Node &N, llvm::StringRef Desc) {
    if (auto Scalar = scalarValue(N, Desc)) {
      unsigned long long Num;
      if (!llvm::getAsUnsignedInteger(**Scalar, 0, Num)) {
        return Located<uint32_t>(Num, Scalar->Range);
      }
    }
    warning(Desc + " invalid number", N);
    return std::nullopt;
  }

  // Try to parse a list of single scalar values, or just a single value.
  std::optional<std::vector<Located<std::string>>> scalarValues(Node &N) {
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
      return std::nullopt;
    }
    return Result;
  }

  // Report a "hard" error, reflecting a config file that can never be valid.
  void error(const llvm::Twine &Msg, llvm::SMRange Range) {
    HadError = true;
    SM.PrintMessage(Range.Start, llvm::SourceMgr::DK_Error, Msg, Range);
  }
  void error(const llvm::Twine &Msg, const Node &N) {
    return error(Msg, N.getSourceRange());
  }

  // Report a "soft" error that could be caused by e.g. version skew.
  void warning(const llvm::Twine &Msg, llvm::SMRange Range) {
    SM.PrintMessage(Range.Start, llvm::SourceMgr::DK_Warning, Msg, Range);
  }
  void warning(const llvm::Twine &Msg, const Node &N) {
    return warning(Msg, N.getSourceRange());
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
      SM->PrintMessage(Fragment.Source.Location, llvm::SourceMgr::DK_Note,
                       "Parsing config fragment");
      if (Parser(*SM).parse(Fragment, *N))
        Result.push_back(std::move(Fragment));
    }
  }
  SM->PrintMessage(SM->FindLocForLineAndColumn(SM->getMainFileID(), 0, 0),
                   llvm::SourceMgr::DK_Note,
                   "Parsed " + llvm::Twine(Result.size()) +
                       " fragments from file");
  // Hack: stash the buffer in the SourceMgr to keep it alive.
  // SM has two entries: "main" non-owning buffer, and ignored owning buffer.
  SM->AddNewSourceBuffer(std::move(Buf), llvm::SMLoc());
  return Result;
}

} // namespace config
} // namespace clangd
} // namespace clang
