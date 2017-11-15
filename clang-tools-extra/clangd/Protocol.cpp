//===--- Protocol.cpp - Language Server Protocol Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the LSP structs.
// FIXME: This is extremely repetetive and ugly. Is there a better way?
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "Logger.h"

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::clangd;

namespace {
void logIgnoredField(llvm::StringRef KeyValue, clangd::Logger &Logger) {
  Logger.log(llvm::formatv("Ignored unknown field \"{0}\"\n", KeyValue));
}
} // namespace

URI URI::fromUri(llvm::StringRef uri) {
  URI Result;
  Result.uri = uri;
  uri.consume_front("file://");
  // Also trim authority-less URIs
  uri.consume_front("file:");
  // For Windows paths e.g. /X:
  if (uri.size() > 2 && uri[0] == '/' && uri[2] == ':')
    uri.consume_front("/");
  // Make sure that file paths are in native separators
  Result.file = llvm::sys::path::convert_to_slash(uri);
  return Result;
}

URI URI::fromFile(llvm::StringRef file) {
  using namespace llvm::sys;
  URI Result;
  Result.file = file;
  Result.uri = "file://";
  // For Windows paths e.g. X:
  if (file.size() > 1 && file[1] == ':')
    Result.uri += "/";
  // Make sure that uri paths are with posix separators
  Result.uri += path::convert_to_slash(file, path::Style::posix);
  return Result;
}

URI URI::parse(llvm::yaml::ScalarNode *Param) {
  llvm::SmallString<10> Storage;
  return URI::fromUri(Param->getValue(Storage));
}

json::Expr URI::unparse(const URI &U) { return U.uri; }

llvm::Optional<TextDocumentIdentifier>
TextDocumentIdentifier::parse(llvm::yaml::MappingNode *Params,
                              clangd::Logger &Logger) {
  TextDocumentIdentifier Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    if (KeyValue == "uri") {
      Result.uri = URI::parse(Value);
    } else if (KeyValue == "version") {
      // FIXME: parse version, but only for VersionedTextDocumentIdentifiers.
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<Position> Position::parse(llvm::yaml::MappingNode *Params,
                                         clangd::Logger &Logger) {
  Position Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "line") {
      long long Val;
      if (llvm::getAsSignedInteger(Value->getValue(Storage), 0, Val))
        return llvm::None;
      Result.line = Val;
    } else if (KeyValue == "character") {
      long long Val;
      if (llvm::getAsSignedInteger(Value->getValue(Storage), 0, Val))
        return llvm::None;
      Result.character = Val;
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

json::Expr Position::unparse(const Position &P) {
  return json::obj{
      {"line", P.line},
      {"character", P.character},
  };
}

llvm::Optional<Range> Range::parse(llvm::yaml::MappingNode *Params,
                                   clangd::Logger &Logger) {
  Range Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "start") {
      auto Parsed = Position::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.start = std::move(*Parsed);
    } else if (KeyValue == "end") {
      auto Parsed = Position::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.end = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

json::Expr Range::unparse(const Range &P) {
  return json::obj{
      {"start", P.start},
      {"end", P.end},
  };
}

json::Expr Location::unparse(const Location &P) {
  return json::obj{
      {"uri", P.uri},
      {"range", P.range},
  };
}

llvm::Optional<TextDocumentItem>
TextDocumentItem::parse(llvm::yaml::MappingNode *Params,
                        clangd::Logger &Logger) {
  TextDocumentItem Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "uri") {
      Result.uri = URI::parse(Value);
    } else if (KeyValue == "languageId") {
      Result.languageId = Value->getValue(Storage);
    } else if (KeyValue == "version") {
      long long Val;
      if (llvm::getAsSignedInteger(Value->getValue(Storage), 0, Val))
        return llvm::None;
      Result.version = Val;
    } else if (KeyValue == "text") {
      Result.text = Value->getValue(Storage);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<Metadata> Metadata::parse(llvm::yaml::MappingNode *Params,
                                         clangd::Logger &Logger) {
  Metadata Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value = NextKeyValue.getValue();

    llvm::SmallString<10> Storage;
    if (KeyValue == "extraFlags") {
      auto *Seq = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Seq)
        return llvm::None;
      for (auto &Item : *Seq) {
        auto *Node = dyn_cast<llvm::yaml::ScalarNode>(&Item);
        if (!Node)
          return llvm::None;
        Result.extraFlags.push_back(Node->getValue(Storage));
      }
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<TextEdit> TextEdit::parse(llvm::yaml::MappingNode *Params,
                                         clangd::Logger &Logger) {
  TextEdit Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value = NextKeyValue.getValue();

    llvm::SmallString<10> Storage;
    if (KeyValue == "range") {
      auto *Map = dyn_cast<llvm::yaml::MappingNode>(Value);
      if (!Map)
        return llvm::None;
      auto Parsed = Range::parse(Map, Logger);
      if (!Parsed)
        return llvm::None;
      Result.range = std::move(*Parsed);
    } else if (KeyValue == "newText") {
      auto *Node = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Node)
        return llvm::None;
      Result.newText = Node->getValue(Storage);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

json::Expr TextEdit::unparse(const TextEdit &P) {
  return json::obj{
      {"range", P.range},
      {"newText", P.newText},
  };
}

namespace {
TraceLevel getTraceLevel(llvm::StringRef TraceLevelStr,
                         clangd::Logger &Logger) {
  if (TraceLevelStr == "off")
    return TraceLevel::Off;
  else if (TraceLevelStr == "messages")
    return TraceLevel::Messages;
  else if (TraceLevelStr == "verbose")
    return TraceLevel::Verbose;

  Logger.log(llvm::formatv("Unknown trace level \"{0}\"\n", TraceLevelStr));
  return TraceLevel::Off;
}
} // namespace

llvm::Optional<InitializeParams>
InitializeParams::parse(llvm::yaml::MappingNode *Params,
                        clangd::Logger &Logger) {
  // If we don't understand the params, proceed with default parameters.
  auto ParseFailure = [&] {
    Logger.log("Failed to decode InitializeParams\n");
    return InitializeParams();
  };
  InitializeParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return ParseFailure();

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
    if (!Value)
      continue;

    if (KeyValue == "processId") {
      auto *Value =
          dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
      if (!Value)
        return ParseFailure();
      long long Val;
      if (llvm::getAsSignedInteger(Value->getValue(KeyStorage), 0, Val))
        return ParseFailure();
      Result.processId = Val;
    } else if (KeyValue == "rootPath") {
      Result.rootPath = Value->getValue(KeyStorage);
    } else if (KeyValue == "rootUri") {
      Result.rootUri = URI::parse(Value);
    } else if (KeyValue == "initializationOptions") {
      // Not used
    } else if (KeyValue == "capabilities") {
      // Not used
    } else if (KeyValue == "trace") {
      Result.trace = getTraceLevel(Value->getValue(KeyStorage), Logger);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<DidOpenTextDocumentParams>
DidOpenTextDocumentParams::parse(llvm::yaml::MappingNode *Params,
                                 clangd::Logger &Logger) {
  DidOpenTextDocumentParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "textDocument") {
      auto Parsed = TextDocumentItem::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else if (KeyValue == "metadata") {
      auto Parsed = Metadata::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.metadata = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<DidCloseTextDocumentParams>
DidCloseTextDocumentParams::parse(llvm::yaml::MappingNode *Params,
                                  clangd::Logger &Logger) {
  DidCloseTextDocumentParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value = NextKeyValue.getValue();

    if (KeyValue == "textDocument") {
      auto *Map = dyn_cast<llvm::yaml::MappingNode>(Value);
      if (!Map)
        return llvm::None;
      auto Parsed = TextDocumentIdentifier::parse(Map, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<DidChangeTextDocumentParams>
DidChangeTextDocumentParams::parse(llvm::yaml::MappingNode *Params,
                                   clangd::Logger &Logger) {
  DidChangeTextDocumentParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value = NextKeyValue.getValue();

    llvm::SmallString<10> Storage;
    if (KeyValue == "textDocument") {
      auto *Map = dyn_cast<llvm::yaml::MappingNode>(Value);
      if (!Map)
        return llvm::None;
      auto Parsed = TextDocumentIdentifier::parse(Map, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else if (KeyValue == "contentChanges") {
      auto *Seq = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Seq)
        return llvm::None;
      for (auto &Item : *Seq) {
        auto *I = dyn_cast<llvm::yaml::MappingNode>(&Item);
        if (!I)
          return llvm::None;
        auto Parsed = TextDocumentContentChangeEvent::parse(I, Logger);
        if (!Parsed)
          return llvm::None;
        Result.contentChanges.push_back(std::move(*Parsed));
      }
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<FileEvent> FileEvent::parse(llvm::yaml::MappingNode *Params,
                                           clangd::Logger &Logger) {
  llvm::Optional<FileEvent> Result = FileEvent();
  for (auto &NextKeyValue : *Params) {
    // We have to consume the whole MappingNode because it doesn't support
    // skipping and we want to be able to parse further valid events.
    if (!Result)
      continue;

    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString) {
      Result.reset();
      continue;
    }

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
    if (!Value) {
      Result.reset();
      continue;
    }
    llvm::SmallString<10> Storage;
    if (KeyValue == "uri") {
      Result->uri = URI::parse(Value);
    } else if (KeyValue == "type") {
      long long Val;
      if (llvm::getAsSignedInteger(Value->getValue(Storage), 0, Val)) {
        Result.reset();
        continue;
      }
      Result->type = static_cast<FileChangeType>(Val);
      if (Result->type < FileChangeType::Created ||
          Result->type > FileChangeType::Deleted)
        Result.reset();
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<DidChangeWatchedFilesParams>
DidChangeWatchedFilesParams::parse(llvm::yaml::MappingNode *Params,
                                   clangd::Logger &Logger) {
  DidChangeWatchedFilesParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value = NextKeyValue.getValue();

    llvm::SmallString<10> Storage;
    if (KeyValue == "changes") {
      auto *Seq = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Seq)
        return llvm::None;
      for (auto &Item : *Seq) {
        auto *I = dyn_cast<llvm::yaml::MappingNode>(&Item);
        if (!I)
          return llvm::None;
        auto Parsed = FileEvent::parse(I, Logger);
        if (Parsed)
          Result.changes.push_back(std::move(*Parsed));
        else
          Logger.log("Failed to decode a FileEvent.\n");
      }
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<TextDocumentContentChangeEvent>
TextDocumentContentChangeEvent::parse(llvm::yaml::MappingNode *Params,
                                      clangd::Logger &Logger) {
  TextDocumentContentChangeEvent Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "text") {
      Result.text = Value->getValue(Storage);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<FormattingOptions>
FormattingOptions::parse(llvm::yaml::MappingNode *Params,
                         clangd::Logger &Logger) {
  FormattingOptions Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "tabSize") {
      long long Val;
      if (llvm::getAsSignedInteger(Value->getValue(Storage), 0, Val))
        return llvm::None;
      Result.tabSize = Val;
    } else if (KeyValue == "insertSpaces") {
      long long Val;
      StringRef Str = Value->getValue(Storage);
      if (llvm::getAsSignedInteger(Str, 0, Val)) {
        if (Str == "true")
          Val = 1;
        else if (Str == "false")
          Val = 0;
        else
          return llvm::None;
      }
      Result.insertSpaces = Val;
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

json::Expr FormattingOptions::unparse(const FormattingOptions &P) {
  return json::obj{
      {"tabSize", P.tabSize},
      {"insertSpaces", P.insertSpaces},
  };
}

llvm::Optional<DocumentRangeFormattingParams>
DocumentRangeFormattingParams::parse(llvm::yaml::MappingNode *Params,
                                     clangd::Logger &Logger) {
  DocumentRangeFormattingParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "textDocument") {
      auto Parsed = TextDocumentIdentifier::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else if (KeyValue == "range") {
      auto Parsed = Range::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.range = std::move(*Parsed);
    } else if (KeyValue == "options") {
      auto Parsed = FormattingOptions::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.options = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<DocumentOnTypeFormattingParams>
DocumentOnTypeFormattingParams::parse(llvm::yaml::MappingNode *Params,
                                      clangd::Logger &Logger) {
  DocumentOnTypeFormattingParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);

    if (KeyValue == "ch") {
      auto *ScalarValue =
          dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
      if (!ScalarValue)
        return llvm::None;
      llvm::SmallString<10> Storage;
      Result.ch = ScalarValue->getValue(Storage);
      continue;
    }

    auto *Value =
        dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;
    if (KeyValue == "textDocument") {
      auto Parsed = TextDocumentIdentifier::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else if (KeyValue == "position") {
      auto Parsed = Position::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.position = std::move(*Parsed);
    } else if (KeyValue == "options") {
      auto Parsed = FormattingOptions::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.options = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<DocumentFormattingParams>
DocumentFormattingParams::parse(llvm::yaml::MappingNode *Params,
                                clangd::Logger &Logger) {
  DocumentFormattingParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "textDocument") {
      auto Parsed = TextDocumentIdentifier::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else if (KeyValue == "options") {
      auto Parsed = FormattingOptions::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.options = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<Diagnostic> Diagnostic::parse(llvm::yaml::MappingNode *Params,
                                             clangd::Logger &Logger) {
  Diagnostic Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);

    llvm::SmallString<10> Storage;
    if (KeyValue == "range") {
      auto *Value =
          dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
      if (!Value)
        return llvm::None;
      auto Parsed = Range::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.range = std::move(*Parsed);
    } else if (KeyValue == "severity") {
      auto *Value =
          dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
      if (!Value)
        return llvm::None;
      long long Val;
      if (llvm::getAsSignedInteger(Value->getValue(Storage), 0, Val))
        return llvm::None;
      Result.severity = Val;
    } else if (KeyValue == "code") {
      // Not currently used
    } else if (KeyValue == "source") {
      // Not currently used
    } else if (KeyValue == "message") {
      auto *Value =
          dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
      if (!Value)
        return llvm::None;
      Result.message = Value->getValue(Storage);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<CodeActionContext>
CodeActionContext::parse(llvm::yaml::MappingNode *Params,
                         clangd::Logger &Logger) {
  CodeActionContext Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value = NextKeyValue.getValue();

    llvm::SmallString<10> Storage;
    if (KeyValue == "diagnostics") {
      auto *Seq = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Seq)
        return llvm::None;
      for (auto &Item : *Seq) {
        auto *I = dyn_cast<llvm::yaml::MappingNode>(&Item);
        if (!I)
          return llvm::None;
        auto Parsed = Diagnostic::parse(I, Logger);
        if (!Parsed)
          return llvm::None;
        Result.diagnostics.push_back(std::move(*Parsed));
      }
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<CodeActionParams>
CodeActionParams::parse(llvm::yaml::MappingNode *Params,
                        clangd::Logger &Logger) {
  CodeActionParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;

    llvm::SmallString<10> Storage;
    if (KeyValue == "textDocument") {
      auto Parsed = TextDocumentIdentifier::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else if (KeyValue == "range") {
      auto Parsed = Range::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.range = std::move(*Parsed);
    } else if (KeyValue == "context") {
      auto Parsed = CodeActionContext::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.context = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

llvm::Optional<std::map<std::string, std::vector<TextEdit>>>
parseWorkspaceEditChange(llvm::yaml::MappingNode *Params,
                         clangd::Logger &Logger) {
  std::map<std::string, std::vector<TextEdit>> Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    if (Result.count(KeyValue)) {
      logIgnoredField(KeyValue, Logger);
      continue;
    }

    auto *Value =
        dyn_cast_or_null<llvm::yaml::SequenceNode>(NextKeyValue.getValue());
    if (!Value)
      return llvm::None;
    for (auto &Item : *Value) {
      auto *ItemValue = dyn_cast_or_null<llvm::yaml::MappingNode>(&Item);
      if (!ItemValue)
        return llvm::None;
      auto Parsed = TextEdit::parse(ItemValue, Logger);
      if (!Parsed)
        return llvm::None;

      Result[KeyValue].push_back(*Parsed);
    }
  }

  return Result;
}

llvm::Optional<WorkspaceEdit>
WorkspaceEdit::parse(llvm::yaml::MappingNode *Params, clangd::Logger &Logger) {
  WorkspaceEdit Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);

    llvm::SmallString<10> Storage;
    if (KeyValue == "changes") {
      auto *Value =
          dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
      if (!Value)
        return llvm::None;
      auto Parsed = parseWorkspaceEditChange(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.changes = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

const std::string ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND =
    "clangd.applyFix";

llvm::Optional<ExecuteCommandParams>
ExecuteCommandParams::parse(llvm::yaml::MappingNode *Params,
                            clangd::Logger &Logger) {
  ExecuteCommandParams Result;
  // Depending on which "command" we parse, we will use this function to parse
  // the command "arguments".
  std::function<bool(llvm::yaml::MappingNode * Params)> ArgParser = nullptr;

  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);

    // Note that "commands" has to be parsed before "arguments" for this to
    // work properly.
    if (KeyValue == "command") {
      auto *ScalarValue =
          dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getValue());
      if (!ScalarValue)
        return llvm::None;
      llvm::SmallString<10> Storage;
      Result.command = ScalarValue->getValue(Storage);
      if (Result.command == ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND) {
        ArgParser = [&Result, &Logger](llvm::yaml::MappingNode *Params) {
          auto WE = WorkspaceEdit::parse(Params, Logger);
          if (WE)
            Result.workspaceEdit = WE;
          return WE.hasValue();
        };
      } else {
        return llvm::None;
      }
    } else if (KeyValue == "arguments") {
      auto *Value = NextKeyValue.getValue();
      auto *Seq = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Seq)
        return llvm::None;
      for (auto &Item : *Seq) {
        auto *ItemValue = dyn_cast_or_null<llvm::yaml::MappingNode>(&Item);
        if (!ItemValue || !ArgParser)
          return llvm::None;
        if (!ArgParser(ItemValue))
          return llvm::None;
      }
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  if (Result.command.empty())
    return llvm::None;

  return Result;
}

json::Expr WorkspaceEdit::unparse(const WorkspaceEdit &WE) {
  if (!WE.changes)
    return json::obj{};
  json::obj FileChanges;
  for (auto &Change : *WE.changes)
    FileChanges[Change.first] = json::ary(Change.second);
  return json::obj{{"changes", std::move(FileChanges)}};
}

json::Expr
ApplyWorkspaceEditParams::unparse(const ApplyWorkspaceEditParams &Params) {
  return json::obj{{"edit", Params.edit}};
}

llvm::Optional<TextDocumentPositionParams>
TextDocumentPositionParams::parse(llvm::yaml::MappingNode *Params,
                                  clangd::Logger &Logger) {
  TextDocumentPositionParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    auto *Value =
        dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
    if (!Value)
      continue;

    llvm::SmallString<10> Storage;
    if (KeyValue == "textDocument") {
      auto Parsed = TextDocumentIdentifier::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else if (KeyValue == "position") {
      auto Parsed = Position::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.position = std::move(*Parsed);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}

json::Expr CompletionItem::unparse(const CompletionItem &CI) {
  assert(!CI.label.empty() && "completion item label is required");
  json::obj Result{{"label", CI.label}};
  if (CI.kind != CompletionItemKind::Missing)
    Result["kind"] = static_cast<int>(CI.kind);
  if (!CI.detail.empty())
    Result["detail"] = CI.detail;
  if (!CI.documentation.empty())
    Result["documentation"] = CI.documentation;
  if (!CI.sortText.empty())
    Result["sortText"] = CI.sortText;
  if (!CI.filterText.empty())
    Result["filterText"] = CI.filterText;
  if (!CI.insertText.empty())
    Result["insertText"] = CI.insertText;
  if (CI.insertTextFormat != InsertTextFormat::Missing)
    Result["insertTextFormat"] = static_cast<int>(CI.insertTextFormat);
  if (CI.textEdit)
    Result["textEdit"] = *CI.textEdit;
  if (!CI.additionalTextEdits.empty())
    Result["additionalTextEdits"] = json::ary(CI.additionalTextEdits);
  return std::move(Result);
}

bool clangd::operator<(const CompletionItem &L, const CompletionItem &R) {
  return (L.sortText.empty() ? L.label : L.sortText) <
         (R.sortText.empty() ? R.label : R.sortText);
}

json::Expr CompletionList::unparse(const CompletionList &L) {
  return json::obj{
      {"isIncomplete", L.isIncomplete},
      {"items", json::ary(L.items)},
  };
}

json::Expr ParameterInformation::unparse(const ParameterInformation &PI) {
  assert(!PI.label.empty() && "parameter information label is required");
  json::obj Result{{"label", PI.label}};
  if (!PI.documentation.empty())
    Result["documentation"] = PI.documentation;
  return std::move(Result);
}

json::Expr SignatureInformation::unparse(const SignatureInformation &SI) {
  assert(!SI.label.empty() && "signature information label is required");
  json::obj Result{
      {"label", SI.label},
      {"parameters", json::ary(SI.parameters)},
  };
  if (!SI.documentation.empty())
    Result["documentation"] = SI.documentation;
  return std::move(Result);
}

json::Expr SignatureHelp::unparse(const SignatureHelp &SH) {
  assert(SH.activeSignature >= 0 &&
         "Unexpected negative value for number of active signatures.");
  assert(SH.activeParameter >= 0 &&
         "Unexpected negative value for active parameter index");
  return json::obj{
      {"activeSignature", SH.activeSignature},
      {"activeParameter", SH.activeParameter},
      {"signatures", json::ary(SH.signatures)},
  };
}

llvm::Optional<RenameParams>
RenameParams::parse(llvm::yaml::MappingNode *Params, clangd::Logger &Logger) {
  RenameParams Result;
  for (auto &NextKeyValue : *Params) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return llvm::None;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);

    if (KeyValue == "textDocument") {
      auto *Value =
        dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
      if (!Value)
        continue;
      auto *Map = dyn_cast<llvm::yaml::MappingNode>(Value);
      if (!Map)
        return llvm::None;
      auto Parsed = TextDocumentIdentifier::parse(Map, Logger);
      if (!Parsed)
        return llvm::None;
      Result.textDocument = std::move(*Parsed);
    } else if (KeyValue == "position") {
      auto *Value =
          dyn_cast_or_null<llvm::yaml::MappingNode>(NextKeyValue.getValue());
      if (!Value)
        continue;
      auto Parsed = Position::parse(Value, Logger);
      if (!Parsed)
        return llvm::None;
      Result.position = std::move(*Parsed);
    } else if (KeyValue == "newName") {
      auto *Value = NextKeyValue.getValue();
      if (!Value)
        continue;
      auto *Node = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Node)
        return llvm::None;
      llvm::SmallString<10> Storage;
      Result.newName = Node->getValue(Storage);
    } else {
      logIgnoredField(KeyValue, Logger);
    }
  }
  return Result;
}
