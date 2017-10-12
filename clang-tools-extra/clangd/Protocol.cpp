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

std::string URI::unparse(const URI &U) { return "\"" + U.uri + "\""; }

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

std::string Position::unparse(const Position &P) {
  std::string Result;
  llvm::raw_string_ostream(Result)
      << llvm::format(R"({"line": %d, "character": %d})", P.line, P.character);
  return Result;
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

std::string Range::unparse(const Range &P) {
  std::string Result;
  llvm::raw_string_ostream(Result) << llvm::format(
      R"({"start": %s, "end": %s})", Position::unparse(P.start).c_str(),
      Position::unparse(P.end).c_str());
  return Result;
}

std::string Location::unparse(const Location &P) {
  std::string Result;
  llvm::raw_string_ostream(Result) << llvm::format(
      R"({"uri": %s, "range": %s})", URI::unparse(P.uri).c_str(),
      Range::unparse(P.range).c_str());
  return Result;
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

std::string TextEdit::unparse(const TextEdit &P) {
  std::string Result;
  llvm::raw_string_ostream(Result) << llvm::format(
      R"({"range": %s, "newText": "%s"})", Range::unparse(P.range).c_str(),
      llvm::yaml::escape(P.newText).c_str());
  return Result;
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

std::string FormattingOptions::unparse(const FormattingOptions &P) {
  std::string Result;
  llvm::raw_string_ostream(Result) << llvm::format(
      R"({"tabSize": %d, "insertSpaces": %d})", P.tabSize, P.insertSpaces);
  return Result;
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

std::string CompletionItem::unparse(const CompletionItem &CI) {
  std::string Result = "{";
  llvm::raw_string_ostream Os(Result);
  assert(!CI.label.empty() && "completion item label is required");
  Os << R"("label":")" << llvm::yaml::escape(CI.label) << R"(",)";
  if (CI.kind != CompletionItemKind::Missing)
    Os << R"("kind":)" << static_cast<int>(CI.kind) << R"(,)";
  if (!CI.detail.empty())
    Os << R"("detail":")" << llvm::yaml::escape(CI.detail) << R"(",)";
  if (!CI.documentation.empty())
    Os << R"("documentation":")" << llvm::yaml::escape(CI.documentation)
       << R"(",)";
  if (!CI.sortText.empty())
    Os << R"("sortText":")" << llvm::yaml::escape(CI.sortText) << R"(",)";
  if (!CI.filterText.empty())
    Os << R"("filterText":")" << llvm::yaml::escape(CI.filterText) << R"(",)";
  if (!CI.insertText.empty())
    Os << R"("insertText":")" << llvm::yaml::escape(CI.insertText) << R"(",)";
  if (CI.insertTextFormat != InsertTextFormat::Missing) {
    Os << R"("insertTextFormat":)" << static_cast<int>(CI.insertTextFormat)
       << R"(,)";
  }
  if (CI.textEdit)
    Os << R"("textEdit":)" << TextEdit::unparse(*CI.textEdit) << ',';
  if (!CI.additionalTextEdits.empty()) {
    Os << R"("additionalTextEdits":[)";
    for (const auto &Edit : CI.additionalTextEdits)
      Os << TextEdit::unparse(Edit) << ",";
    Os.flush();
    // The list additionalTextEdits is guaranteed nonempty at this point.
    // Replace the trailing comma with right brace.
    Result.back() = ']';
  }
  Os.flush();
  // Label is required, so Result is guaranteed to have a trailing comma.
  Result.back() = '}';
  return Result;
}

std::string ParameterInformation::unparse(const ParameterInformation &PI) {
  std::string Result = "{";
  llvm::raw_string_ostream Os(Result);
  assert(!PI.label.empty() && "parameter information label is required");
  Os << R"("label":")" << llvm::yaml::escape(PI.label) << '\"';
  if (!PI.documentation.empty())
    Os << R"(,"documentation":")" << llvm::yaml::escape(PI.documentation)
       << '\"';
  Os << '}';
  Os.flush();
  return Result;
}

std::string SignatureInformation::unparse(const SignatureInformation &SI) {
  std::string Result = "{";
  llvm::raw_string_ostream Os(Result);
  assert(!SI.label.empty() && "signature information label is required");
  Os << R"("label":")" << llvm::yaml::escape(SI.label) << '\"';
  if (!SI.documentation.empty())
    Os << R"(,"documentation":")" << llvm::yaml::escape(SI.documentation)
       << '\"';
  Os << R"(,"parameters":[)";
  for (const auto &Parameter : SI.parameters) {
    Os << ParameterInformation::unparse(Parameter) << ',';
  }
  Os.flush();
  if (SI.parameters.empty())
    Result.push_back(']');
  else
    Result.back() = ']'; // Replace the last `,` with an `]`.
  Result.push_back('}');
  return Result;
}

std::string SignatureHelp::unparse(const SignatureHelp &SH) {
  std::string Result = "{";
  llvm::raw_string_ostream Os(Result);
  assert(SH.activeSignature >= 0 &&
         "Unexpected negative value for number of active signatures.");
  assert(SH.activeParameter >= 0 &&
         "Unexpected negative value for active parameter index");
  Os << R"("activeSignature":)" << SH.activeSignature
     << R"(,"activeParameter":)" << SH.activeParameter << R"(,"signatures":[)";
  for (const auto &Signature : SH.signatures) {
    Os << SignatureInformation::unparse(Signature) << ',';
  }
  Os.flush();
  if (SH.signatures.empty())
    Result.push_back(']');
  else
    Result.back() = ']'; // Replace the last `,` with an `]`.
  Result.push_back('}');
  return Result;
}
