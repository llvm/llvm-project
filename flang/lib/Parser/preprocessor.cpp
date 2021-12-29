//===-- lib/Parser/preprocessor.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "preprocessor.h"
#include "prescan.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <ctime>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>

namespace Fortran::parser {

Definition::Definition(
    const TokenSequence &repl, std::size_t firstToken, std::size_t tokens)
    : replacement_{Tokenize({}, repl, firstToken, tokens)} {}

Definition::Definition(const std::vector<std::string> &argNames,
    const TokenSequence &repl, std::size_t firstToken, std::size_t tokens,
    bool isVariadic)
    : isFunctionLike_{true},
      argumentCount_(argNames.size()), isVariadic_{isVariadic},
      replacement_{Tokenize(argNames, repl, firstToken, tokens)} {}

Definition::Definition(const std::string &predefined, AllSources &sources)
    : isPredefined_{true},
      replacement_{
          predefined, sources.AddCompilerInsertion(predefined).start()} {}

bool Definition::set_isDisabled(bool disable) {
  bool was{isDisabled_};
  isDisabled_ = disable;
  return was;
}

static bool IsLegalIdentifierStart(const CharBlock &cpl) {
  return cpl.size() > 0 && IsLegalIdentifierStart(cpl[0]);
}

TokenSequence Definition::Tokenize(const std::vector<std::string> &argNames,
    const TokenSequence &token, std::size_t firstToken, std::size_t tokens) {
  std::map<std::string, std::string> args;
  char argIndex{'A'};
  for (const std::string &arg : argNames) {
    CHECK(args.find(arg) == args.end());
    args[arg] = "~"s + argIndex++;
  }
  TokenSequence result;
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock tok{token.TokenAt(firstToken + j)};
    if (IsLegalIdentifierStart(tok)) {
      auto it{args.find(tok.ToString())};
      if (it != args.end()) {
        result.Put(it->second, token.GetTokenProvenance(j));
        continue;
      }
    }
    result.Put(token, firstToken + j, 1);
  }
  return result;
}

static TokenSequence Stringify(
    const TokenSequence &tokens, AllSources &allSources) {
  TokenSequence result;
  Provenance quoteProvenance{allSources.CompilerInsertionProvenance('"')};
  result.PutNextTokenChar('"', quoteProvenance);
  for (std::size_t j{0}; j < tokens.SizeInTokens(); ++j) {
    const CharBlock &token{tokens.TokenAt(j)};
    std::size_t bytes{token.size()};
    for (std::size_t k{0}; k < bytes; ++k) {
      char ch{token[k]};
      Provenance from{tokens.GetTokenProvenance(j, k)};
      if (ch == '"' || ch == '\\') {
        result.PutNextTokenChar(ch, from);
      }
      result.PutNextTokenChar(ch, from);
    }
  }
  result.PutNextTokenChar('"', quoteProvenance);
  result.CloseToken();
  return result;
}

constexpr bool IsTokenPasting(CharBlock opr) {
  return opr.size() == 2 && opr[0] == '#' && opr[1] == '#';
}

static bool AnyTokenPasting(const TokenSequence &text) {
  std::size_t tokens{text.SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    if (IsTokenPasting(text.TokenAt(j))) {
      return true;
    }
  }
  return false;
}

static TokenSequence TokenPasting(TokenSequence &&text) {
  if (!AnyTokenPasting(text)) {
    return std::move(text);
  }
  TokenSequence result;
  std::size_t tokens{text.SizeInTokens()};
  bool pasting{false};
  for (std::size_t j{0}; j < tokens; ++j) {
    if (IsTokenPasting(text.TokenAt(j))) {
      if (!pasting) {
        while (!result.empty() &&
            result.TokenAt(result.SizeInTokens() - 1).IsBlank()) {
          result.pop_back();
        }
        if (!result.empty()) {
          result.ReopenLastToken();
          pasting = true;
        }
      }
    } else if (pasting && text.TokenAt(j).IsBlank()) {
    } else {
      result.Put(text, j, 1);
      pasting = false;
    }
  }
  return result;
}

TokenSequence Definition::Apply(
    const std::vector<TokenSequence> &args, Prescanner &prescanner) {
  TokenSequence result;
  bool skipping{false};
  int parenthesesNesting{0};
  std::size_t tokens{replacement_.SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock token{replacement_.TokenAt(j)};
    std::size_t bytes{token.size()};
    if (skipping) {
      if (bytes == 1) {
        if (token[0] == '(') {
          ++parenthesesNesting;
        } else if (token[0] == ')') {
          skipping = --parenthesesNesting > 0;
        }
      }
      continue;
    }
    if (bytes == 2 && token[0] == '~') { // argument substitution
      std::size_t index = token[1] - 'A';
      if (index >= args.size()) {
        continue;
      }
      std::size_t prev{j};
      while (prev > 0 && replacement_.TokenAt(prev - 1).IsBlank()) {
        --prev;
      }
      if (prev > 0 && replacement_.TokenAt(prev - 1).size() == 1 &&
          replacement_.TokenAt(prev - 1)[0] ==
              '#') { // stringify argument without macro replacement
        std::size_t resultSize{result.SizeInTokens()};
        while (resultSize > 0 && result.TokenAt(resultSize - 1).IsBlank()) {
          result.pop_back();
          --resultSize;
        }
        CHECK(resultSize > 0 &&
            result.TokenAt(resultSize - 1) == replacement_.TokenAt(prev - 1));
        result.pop_back();
        result.Put(Stringify(args[index], prescanner.allSources()));
      } else {
        const TokenSequence *arg{&args[index]};
        std::optional<TokenSequence> replaced;
        // Don't replace macros in the actual argument if it is preceded or
        // followed by the token-pasting operator ## in the replacement text.
        if (prev == 0 || !IsTokenPasting(replacement_.TokenAt(prev - 1))) {
          auto next{replacement_.SkipBlanks(j + 1)};
          if (next >= tokens || !IsTokenPasting(replacement_.TokenAt(next))) {
            // Apply macro replacement to the actual argument
            replaced =
                prescanner.preprocessor().MacroReplacement(*arg, prescanner);
            if (replaced) {
              arg = &*replaced;
            }
          }
        }
        result.Put(DEREF(arg));
      }
    } else if (bytes == 11 && isVariadic_ &&
        token.ToString() == "__VA_ARGS__") {
      Provenance commaProvenance{
          prescanner.preprocessor().allSources().CompilerInsertionProvenance(
              ',')};
      for (std::size_t k{argumentCount_}; k < args.size(); ++k) {
        if (k > argumentCount_) {
          result.Put(","s, commaProvenance);
        }
        result.Put(args[k]);
      }
    } else if (bytes == 10 && isVariadic_ && token.ToString() == "__VA_OPT__" &&
        j + 2 < tokens && replacement_.TokenAt(j + 1).ToString() == "(" &&
        parenthesesNesting == 0) {
      parenthesesNesting = 1;
      skipping = args.size() == argumentCount_;
      ++j;
    } else {
      if (bytes == 1 && parenthesesNesting > 0 && token[0] == '(') {
        ++parenthesesNesting;
      } else if (bytes == 1 && parenthesesNesting > 0 && token[0] == ')') {
        if (--parenthesesNesting == 0) {
          skipping = false;
          continue;
        }
      }
      result.Put(replacement_, j);
    }
  }
  return TokenPasting(std::move(result));
}

static std::string FormatTime(const std::time_t &now, const char *format) {
  char buffer[16];
  return {buffer,
      std::strftime(buffer, sizeof buffer, format, std::localtime(&now))};
}

Preprocessor::Preprocessor(AllSources &allSources) : allSources_{allSources} {}

void Preprocessor::DefineStandardMacros() {
  // Capture current local date & time once now to avoid having the values
  // of __DATE__ or __TIME__ change during compilation.
  std::time_t now;
  std::time(&now);
  Define("__DATE__"s, FormatTime(now, "\"%h %e %Y\"")); // e.g., "Jun 16 1904"
  Define("__TIME__"s, FormatTime(now, "\"%T\"")); // e.g., "23:59:60"
  // The values of these predefined macros depend on their invocation sites.
  Define("__FILE__"s, "__FILE__"s);
  Define("__LINE__"s, "__LINE__"s);
}

void Preprocessor::Define(std::string macro, std::string value) {
  definitions_.emplace(SaveTokenAsName(macro), Definition{value, allSources_});
}

void Preprocessor::Undefine(std::string macro) { definitions_.erase(macro); }

std::optional<TokenSequence> Preprocessor::MacroReplacement(
    const TokenSequence &input, Prescanner &prescanner) {
  // Do quick scan for any use of a defined name.
  if (definitions_.empty()) {
    return std::nullopt;
  }
  std::size_t tokens{input.SizeInTokens()};
  std::size_t j;
  for (j = 0; j < tokens; ++j) {
    CharBlock token{input.TokenAt(j)};
    if (!token.empty() && IsLegalIdentifierStart(token[0]) &&
        IsNameDefined(token)) {
      break;
    }
  }
  if (j == tokens) {
    return std::nullopt; // input contains nothing that would be replaced
  }
  TokenSequence result{input, 0, j};
  for (; j < tokens; ++j) {
    const CharBlock &token{input.TokenAt(j)};
    if (token.IsBlank() || !IsLegalIdentifierStart(token[0])) {
      result.Put(input, j);
      continue;
    }
    auto it{definitions_.find(token)};
    if (it == definitions_.end()) {
      result.Put(input, j);
      continue;
    }
    Definition &def{it->second};
    if (def.isDisabled()) {
      result.Put(input, j);
      continue;
    }
    if (!def.isFunctionLike()) {
      if (def.isPredefined()) {
        std::string name{def.replacement().TokenAt(0).ToString()};
        std::string repl;
        if (name == "__FILE__") {
          repl = "\""s +
              allSources_.GetPath(prescanner.GetCurrentProvenance()) + '"';
        } else if (name == "__LINE__") {
          std::string buf;
          llvm::raw_string_ostream ss{buf};
          ss << allSources_.GetLineNumber(prescanner.GetCurrentProvenance());
          repl = ss.str();
        }
        if (!repl.empty()) {
          ProvenanceRange insert{allSources_.AddCompilerInsertion(repl)};
          ProvenanceRange call{allSources_.AddMacroCall(
              insert, input.GetTokenProvenanceRange(j), repl)};
          result.Put(repl, call.start());
          continue;
        }
      }
      def.set_isDisabled(true);
      TokenSequence replaced{
          TokenPasting(ReplaceMacros(def.replacement(), prescanner))};
      def.set_isDisabled(false);
      if (!replaced.empty()) {
        ProvenanceRange from{def.replacement().GetProvenanceRange()};
        ProvenanceRange use{input.GetTokenProvenanceRange(j)};
        ProvenanceRange newRange{
            allSources_.AddMacroCall(from, use, replaced.ToString())};
        result.Put(replaced, newRange);
      }
      continue;
    }
    // Possible function-like macro call.  Skip spaces and newlines to see
    // whether '(' is next.
    std::size_t k{j};
    bool leftParen{false};
    while (++k < tokens) {
      const CharBlock &lookAhead{input.TokenAt(k)};
      if (!lookAhead.IsBlank() && lookAhead[0] != '\n') {
        leftParen = lookAhead[0] == '(' && lookAhead.size() == 1;
        break;
      }
    }
    if (!leftParen) {
      result.Put(input, j);
      continue;
    }
    std::vector<std::size_t> argStart{++k};
    for (int nesting{0}; k < tokens; ++k) {
      CharBlock token{input.TokenAt(k)};
      if (token.size() == 1) {
        char ch{token[0]};
        if (ch == '(') {
          ++nesting;
        } else if (ch == ')') {
          if (nesting == 0) {
            break;
          }
          --nesting;
        } else if (ch == ',' && nesting == 0) {
          argStart.push_back(k + 1);
        }
      }
    }
    if (argStart.size() == 1 && k == argStart[0] && def.argumentCount() == 0) {
      // Subtle: () is zero arguments, not one empty argument,
      // unless one argument was expected.
      argStart.clear();
    }
    if (k >= tokens || argStart.size() < def.argumentCount() ||
        (argStart.size() > def.argumentCount() && !def.isVariadic())) {
      result.Put(input, j);
      continue;
    }
    std::vector<TokenSequence> args;
    for (std::size_t n{0}; n < argStart.size(); ++n) {
      std::size_t at{argStart[n]};
      std::size_t count{
          (n + 1 == argStart.size() ? k : argStart[n + 1] - 1) - at};
      args.emplace_back(TokenSequence(input, at, count));
    }
    def.set_isDisabled(true);
    TokenSequence replaced{
        ReplaceMacros(def.Apply(args, prescanner), prescanner)};
    def.set_isDisabled(false);
    if (!replaced.empty()) {
      ProvenanceRange from{def.replacement().GetProvenanceRange()};
      ProvenanceRange use{input.GetIntervalProvenanceRange(j, k - j)};
      ProvenanceRange newRange{
          allSources_.AddMacroCall(from, use, replaced.ToString())};
      result.Put(replaced, newRange);
    }
    j = k; // advance to the terminal ')'
  }
  return result;
}

TokenSequence Preprocessor::ReplaceMacros(
    const TokenSequence &tokens, Prescanner &prescanner) {
  if (std::optional<TokenSequence> repl{MacroReplacement(tokens, prescanner)}) {
    return std::move(*repl);
  }
  return tokens;
}

void Preprocessor::Directive(const TokenSequence &dir, Prescanner &prescanner) {
  std::size_t tokens{dir.SizeInTokens()};
  std::size_t j{dir.SkipBlanks(0)};
  if (j == tokens) {
    return;
  }
  if (dir.TokenAt(j).ToString() != "#") {
    prescanner.Say(dir.GetTokenProvenanceRange(j), "missing '#'"_err_en_US);
    return;
  }
  j = dir.SkipBlanks(j + 1);
  while (tokens > 0 && dir.TokenAt(tokens - 1).IsBlank()) {
    --tokens;
  }
  if (j == tokens) {
    return;
  }
  if (IsDecimalDigit(dir.TokenAt(j)[0]) || dir.TokenAt(j)[0] == '"') {
    return; // treat like #line, ignore it
  }
  std::size_t dirOffset{j};
  std::string dirName{ToLowerCaseLetters(dir.TokenAt(dirOffset).ToString())};
  j = dir.SkipBlanks(j + 1);
  CharBlock nameToken;
  if (j < tokens && IsLegalIdentifierStart(dir.TokenAt(j)[0])) {
    nameToken = dir.TokenAt(j);
  }
  if (dirName == "line") {
    // #line is ignored
  } else if (dirName == "define") {
    if (nameToken.empty()) {
      prescanner.Say(dir.GetTokenProvenanceRange(j < tokens ? j : tokens - 1),
          "#define: missing or invalid name"_err_en_US);
      return;
    }
    nameToken = SaveTokenAsName(nameToken);
    definitions_.erase(nameToken);
    if (++j < tokens && dir.TokenAt(j).size() == 1 &&
        dir.TokenAt(j)[0] == '(') {
      j = dir.SkipBlanks(j + 1);
      std::vector<std::string> argName;
      bool isVariadic{false};
      if (dir.TokenAt(j).ToString() != ")") {
        while (true) {
          std::string an{dir.TokenAt(j).ToString()};
          if (an == "...") {
            isVariadic = true;
          } else {
            if (an.empty() || !IsLegalIdentifierStart(an[0])) {
              prescanner.Say(dir.GetTokenProvenanceRange(j),
                  "#define: missing or invalid argument name"_err_en_US);
              return;
            }
            argName.push_back(an);
          }
          j = dir.SkipBlanks(j + 1);
          if (j == tokens) {
            prescanner.Say(dir.GetTokenProvenanceRange(tokens - 1),
                "#define: malformed argument list"_err_en_US);
            return;
          }
          std::string punc{dir.TokenAt(j).ToString()};
          if (punc == ")") {
            break;
          }
          if (isVariadic || punc != ",") {
            prescanner.Say(dir.GetTokenProvenanceRange(j),
                "#define: malformed argument list"_err_en_US);
            return;
          }
          j = dir.SkipBlanks(j + 1);
          if (j == tokens) {
            prescanner.Say(dir.GetTokenProvenanceRange(tokens - 1),
                "#define: malformed argument list"_err_en_US);
            return;
          }
        }
        if (std::set<std::string>(argName.begin(), argName.end()).size() !=
            argName.size()) {
          prescanner.Say(dir.GetTokenProvenance(dirOffset),
              "#define: argument names are not distinct"_err_en_US);
          return;
        }
      }
      j = dir.SkipBlanks(j + 1);
      definitions_.emplace(std::make_pair(
          nameToken, Definition{argName, dir, j, tokens - j, isVariadic}));
    } else {
      j = dir.SkipBlanks(j + 1);
      definitions_.emplace(
          std::make_pair(nameToken, Definition{dir, j, tokens - j}));
    }
  } else if (dirName == "undef") {
    if (nameToken.empty()) {
      prescanner.Say(
          dir.GetIntervalProvenanceRange(dirOffset, tokens - dirOffset),
          "# missing or invalid name"_err_en_US);
    } else {
      if (dir.IsAnythingLeft(++j)) {
        prescanner.Say(dir.GetIntervalProvenanceRange(j, tokens - j),
            "#undef: excess tokens at end of directive"_en_US);
      } else {
        definitions_.erase(nameToken);
      }
    }
  } else if (dirName == "ifdef" || dirName == "ifndef") {
    bool doThen{false};
    if (nameToken.empty()) {
      prescanner.Say(
          dir.GetIntervalProvenanceRange(dirOffset, tokens - dirOffset),
          "#%s: missing name"_err_en_US, dirName);
    } else {
      if (dir.IsAnythingLeft(++j)) {
        prescanner.Say(dir.GetIntervalProvenanceRange(j, tokens - j),
            "#%s: excess tokens at end of directive"_en_US, dirName);
      }
      doThen = IsNameDefined(nameToken) == (dirName == "ifdef");
    }
    if (doThen) {
      ifStack_.push(CanDeadElseAppear::Yes);
    } else {
      SkipDisabledConditionalCode(dirName, IsElseActive::Yes, prescanner,
          dir.GetTokenProvenance(dirOffset));
    }
  } else if (dirName == "if") {
    if (IsIfPredicateTrue(dir, j, tokens - j, prescanner)) {
      ifStack_.push(CanDeadElseAppear::Yes);
    } else {
      SkipDisabledConditionalCode(dirName, IsElseActive::Yes, prescanner,
          dir.GetTokenProvenanceRange(dirOffset));
    }
  } else if (dirName == "else") {
    if (dir.IsAnythingLeft(j)) {
      prescanner.Say(dir.GetIntervalProvenanceRange(j, tokens - j),
          "#else: excess tokens at end of directive"_en_US);
    } else if (ifStack_.empty()) {
      prescanner.Say(dir.GetTokenProvenanceRange(dirOffset),
          "#else: not nested within #if, #ifdef, or #ifndef"_err_en_US);
    } else if (ifStack_.top() != CanDeadElseAppear::Yes) {
      prescanner.Say(dir.GetTokenProvenanceRange(dirOffset),
          "#else: already appeared within this #if, #ifdef, or #ifndef"_err_en_US);
    } else {
      ifStack_.pop();
      SkipDisabledConditionalCode("else", IsElseActive::No, prescanner,
          dir.GetTokenProvenanceRange(dirOffset));
    }
  } else if (dirName == "elif") {
    if (ifStack_.empty()) {
      prescanner.Say(dir.GetTokenProvenanceRange(dirOffset),
          "#elif: not nested within #if, #ifdef, or #ifndef"_err_en_US);
    } else if (ifStack_.top() != CanDeadElseAppear::Yes) {
      prescanner.Say(dir.GetTokenProvenanceRange(dirOffset),
          "#elif: #else previously appeared within this #if, #ifdef, or #ifndef"_err_en_US);
    } else {
      ifStack_.pop();
      SkipDisabledConditionalCode("elif", IsElseActive::No, prescanner,
          dir.GetTokenProvenanceRange(dirOffset));
    }
  } else if (dirName == "endif") {
    if (dir.IsAnythingLeft(j)) {
      prescanner.Say(dir.GetIntervalProvenanceRange(j, tokens - j),
          "#endif: excess tokens at end of directive"_en_US);
    } else if (ifStack_.empty()) {
      prescanner.Say(dir.GetTokenProvenanceRange(dirOffset),
          "#endif: no #if, #ifdef, or #ifndef"_err_en_US);
    } else {
      ifStack_.pop();
    }
  } else if (dirName == "error") {
    prescanner.Say(
        dir.GetIntervalProvenanceRange(dirOffset, tokens - dirOffset),
        "%s"_err_en_US, dir.ToString());
  } else if (dirName == "warning" || dirName == "comment" ||
      dirName == "note") {
    prescanner.Say(
        dir.GetIntervalProvenanceRange(dirOffset, tokens - dirOffset),
        "%s"_en_US, dir.ToString());
  } else if (dirName == "include") {
    if (j == tokens) {
      prescanner.Say(
          dir.GetIntervalProvenanceRange(dirOffset, tokens - dirOffset),
          "#include: missing name of file to include"_err_en_US);
      return;
    }
    std::string include;
    std::optional<std::string> prependPath;
    if (dir.TokenAt(j).ToString() == "<") { // #include <foo>
      std::size_t k{j + 1};
      if (k >= tokens) {
        prescanner.Say(dir.GetIntervalProvenanceRange(j, tokens - j),
            "#include: file name missing"_err_en_US);
        return;
      }
      while (k < tokens && dir.TokenAt(k) != ">") {
        ++k;
      }
      if (k >= tokens) {
        prescanner.Say(dir.GetIntervalProvenanceRange(j, tokens - j),
            "#include: expected '>' at end of included file"_en_US);
      }
      TokenSequence braced{dir, j + 1, k - j - 1};
      include = ReplaceMacros(braced, prescanner).ToString();
      j = k;
    } else if ((include = dir.TokenAt(j).ToString()).substr(0, 1) == "\"" &&
        include.substr(include.size() - 1, 1) == "\"") { // #include "foo"
      include = include.substr(1, include.size() - 2);
      // #include "foo" starts search in directory of file containing
      // the directive
      auto prov{dir.GetTokenProvenanceRange(dirOffset).start()};
      if (const auto *currentFile{allSources_.GetSourceFile(prov)}) {
        prependPath = DirectoryName(currentFile->path());
      }
    } else {
      prescanner.Say(dir.GetTokenProvenanceRange(j < tokens ? j : tokens - 1),
          "#include: expected name of file to include"_err_en_US);
      return;
    }
    if (include.empty()) {
      prescanner.Say(dir.GetTokenProvenanceRange(dirOffset),
          "#include: empty include file name"_err_en_US);
      return;
    }
    j = dir.SkipBlanks(j + 1);
    if (j < tokens && dir.TokenAt(j).ToString() != "!") {
      prescanner.Say(dir.GetIntervalProvenanceRange(j, tokens - j),
          "#include: extra stuff ignored after file name"_en_US);
    }
    std::string buf;
    llvm::raw_string_ostream error{buf};
    const SourceFile *included{
        allSources_.Open(include, error, std::move(prependPath))};
    if (!included) {
      prescanner.Say(dir.GetTokenProvenanceRange(dirOffset),
          "#include: %s"_err_en_US, error.str());
    } else if (included->bytes() > 0) {
      ProvenanceRange fileRange{
          allSources_.AddIncludedFile(*included, dir.GetProvenanceRange())};
      Prescanner{prescanner}
          .set_encoding(included->encoding())
          .Prescan(fileRange);
    }
  } else {
    prescanner.Say(dir.GetTokenProvenanceRange(dirOffset),
        "#%s: unknown or unimplemented directive"_err_en_US, dirName);
  }
}

CharBlock Preprocessor::SaveTokenAsName(const CharBlock &t) {
  names_.push_back(t.ToString());
  return {names_.back().data(), names_.back().size()};
}

bool Preprocessor::IsNameDefined(const CharBlock &token) {
  return definitions_.find(token) != definitions_.end();
}

static std::string GetDirectiveName(
    const TokenSequence &line, std::size_t *rest) {
  std::size_t tokens{line.SizeInTokens()};
  std::size_t j{line.SkipBlanks(0)};
  if (j == tokens || line.TokenAt(j).ToString() != "#") {
    *rest = tokens;
    return "";
  }
  j = line.SkipBlanks(j + 1);
  if (j == tokens) {
    *rest = tokens;
    return "";
  }
  *rest = line.SkipBlanks(j + 1);
  return ToLowerCaseLetters(line.TokenAt(j).ToString());
}

void Preprocessor::SkipDisabledConditionalCode(const std::string &dirName,
    IsElseActive isElseActive, Prescanner &prescanner,
    ProvenanceRange provenanceRange) {
  int nesting{0};
  while (!prescanner.IsAtEnd()) {
    if (!prescanner.IsNextLinePreprocessorDirective()) {
      prescanner.NextLine();
      continue;
    }
    TokenSequence line{prescanner.TokenizePreprocessorDirective()};
    std::size_t rest{0};
    std::string dn{GetDirectiveName(line, &rest)};
    if (dn == "ifdef" || dn == "ifndef" || dn == "if") {
      ++nesting;
    } else if (dn == "endif") {
      if (nesting-- == 0) {
        return;
      }
    } else if (isElseActive == IsElseActive::Yes && nesting == 0) {
      if (dn == "else") {
        ifStack_.push(CanDeadElseAppear::No);
        return;
      }
      if (dn == "elif" &&
          IsIfPredicateTrue(
              line, rest, line.SizeInTokens() - rest, prescanner)) {
        ifStack_.push(CanDeadElseAppear::Yes);
        return;
      }
    }
  }
  prescanner.Say(provenanceRange, "#%s: missing #endif"_err_en_US, dirName);
}

// Precedence level codes used here to accommodate mixed Fortran and C:
// 15: parentheses and constants, logical !, bitwise ~
// 14: unary + and -
// 13: **
// 12: *, /, % (modulus)
// 11: + and -
// 10: << and >>
//  9: bitwise &
//  8: bitwise ^
//  7: bitwise |
//  6: relations (.EQ., ==, &c.)
//  5: .NOT.
//  4: .AND., &&
//  3: .OR., ||
//  2: .EQV. and .NEQV. / .XOR.
//  1: ? :
//  0: ,
static std::int64_t ExpressionValue(const TokenSequence &token,
    int minimumPrecedence, std::size_t *atToken,
    std::optional<Message> *error) {
  enum Operator {
    PARENS,
    CONST,
    NOTZERO, // !
    COMPLEMENT, // ~
    UPLUS,
    UMINUS,
    POWER,
    TIMES,
    DIVIDE,
    MODULUS,
    ADD,
    SUBTRACT,
    LEFTSHIFT,
    RIGHTSHIFT,
    BITAND,
    BITXOR,
    BITOR,
    LT,
    LE,
    EQ,
    NE,
    GE,
    GT,
    NOT,
    AND,
    OR,
    EQV,
    NEQV,
    SELECT,
    COMMA
  };
  static const int precedence[]{
      15, 15, 15, 15, // (), 6, !, ~
      14, 14, // unary +, -
      13, 12, 12, 12, 11, 11, 10, 10, // **, *, /, %, +, -, <<, >>
      9, 8, 7, // &, ^, |
      6, 6, 6, 6, 6, 6, // relations .LT. to .GT.
      5, 4, 3, 2, 2, // .NOT., .AND., .OR., .EQV., .NEQV.
      1, 0 // ?: and ,
  };
  static const int operandPrecedence[]{0, -1, 15, 15, 15, 15, 13, 12, 12, 12,
      11, 11, 11, 11, 9, 8, 7, 7, 7, 7, 7, 7, 7, 6, 4, 3, 3, 3, 1, 0};

  static std::map<std::string, enum Operator> opNameMap;
  if (opNameMap.empty()) {
    opNameMap["("] = PARENS;
    opNameMap["!"] = NOTZERO;
    opNameMap["~"] = COMPLEMENT;
    opNameMap["**"] = POWER;
    opNameMap["*"] = TIMES;
    opNameMap["/"] = DIVIDE;
    opNameMap["%"] = MODULUS;
    opNameMap["+"] = ADD;
    opNameMap["-"] = SUBTRACT;
    opNameMap["<<"] = LEFTSHIFT;
    opNameMap[">>"] = RIGHTSHIFT;
    opNameMap["&"] = BITAND;
    opNameMap["^"] = BITXOR;
    opNameMap["|"] = BITOR;
    opNameMap[".lt."] = opNameMap["<"] = LT;
    opNameMap[".le."] = opNameMap["<="] = LE;
    opNameMap[".eq."] = opNameMap["=="] = EQ;
    opNameMap[".ne."] = opNameMap["/="] = opNameMap["!="] = NE;
    opNameMap[".ge."] = opNameMap[">="] = GE;
    opNameMap[".gt."] = opNameMap[">"] = GT;
    opNameMap[".not."] = NOT;
    opNameMap[".and."] = opNameMap[".a."] = opNameMap["&&"] = AND;
    opNameMap[".or."] = opNameMap[".o."] = opNameMap["||"] = OR;
    opNameMap[".eqv."] = EQV;
    opNameMap[".neqv."] = opNameMap[".xor."] = opNameMap[".x."] = NEQV;
    opNameMap["?"] = SELECT;
    opNameMap[","] = COMMA;
  }

  std::size_t tokens{token.SizeInTokens()};
  CHECK(tokens > 0);
  if (*atToken >= tokens) {
    *error =
        Message{token.GetProvenanceRange(), "incomplete expression"_err_en_US};
    return 0;
  }

  // Parse and evaluate a primary or a unary operator and its operand.
  std::size_t opAt{*atToken};
  std::string t{token.TokenAt(opAt).ToString()};
  enum Operator op;
  std::int64_t left{0};
  if (t == "(") {
    op = PARENS;
  } else if (IsDecimalDigit(t[0])) {
    op = CONST;
    std::size_t consumed{0};
    left = std::stoll(t, &consumed, 0 /*base to be detected*/);
    if (consumed < t.size()) {
      *error = Message{token.GetTokenProvenanceRange(opAt),
          "Uninterpretable numeric constant '%s'"_err_en_US, t};
      return 0;
    }
  } else if (IsLegalIdentifierStart(t[0])) {
    // undefined macro name -> zero
    // TODO: BOZ constants?
    op = CONST;
  } else if (t == "+") {
    op = UPLUS;
  } else if (t == "-") {
    op = UMINUS;
  } else if (t == "." && *atToken + 2 < tokens &&
      ToLowerCaseLetters(token.TokenAt(*atToken + 1).ToString()) == "not" &&
      token.TokenAt(*atToken + 2).ToString() == ".") {
    op = NOT;
    *atToken += 2;
  } else {
    auto it{opNameMap.find(t)};
    if (it != opNameMap.end()) {
      op = it->second;
    } else {
      *error = Message{token.GetTokenProvenanceRange(opAt),
          "operand expected in expression"_err_en_US};
      return 0;
    }
  }
  if (precedence[op] < minimumPrecedence) {
    *error = Message{token.GetTokenProvenanceRange(opAt),
        "operator precedence error"_err_en_US};
    return 0;
  }
  ++*atToken;
  if (op != CONST) {
    left = ExpressionValue(token, operandPrecedence[op], atToken, error);
    if (*error) {
      return 0;
    }
    switch (op) {
    case PARENS:
      if (*atToken < tokens && token.TokenAt(*atToken).ToString() == ")") {
        ++*atToken;
        break;
      }
      if (*atToken >= tokens) {
        *error = Message{token.GetProvenanceRange(),
            "')' missing from expression"_err_en_US};
      } else {
        *error = Message{
            token.GetTokenProvenanceRange(*atToken), "expected ')'"_err_en_US};
      }
      return 0;
    case NOTZERO:
      left = !left;
      break;
    case COMPLEMENT:
      left = ~left;
      break;
    case UPLUS:
      break;
    case UMINUS:
      left = -left;
      break;
    case NOT:
      left = -!left;
      break;
    default:
      CRASH_NO_CASE;
    }
  }

  // Parse and evaluate binary operators and their second operands, if present.
  while (*atToken < tokens) {
    int advance{1};
    t = token.TokenAt(*atToken).ToString();
    if (t == "." && *atToken + 2 < tokens &&
        token.TokenAt(*atToken + 2).ToString() == ".") {
      t += ToLowerCaseLetters(token.TokenAt(*atToken + 1).ToString()) + '.';
      advance = 3;
    }
    auto it{opNameMap.find(t)};
    if (it == opNameMap.end()) {
      break;
    }
    op = it->second;
    if (op < POWER || precedence[op] < minimumPrecedence) {
      break;
    }
    opAt = *atToken;
    *atToken += advance;

    std::int64_t right{
        ExpressionValue(token, operandPrecedence[op], atToken, error)};
    if (*error) {
      return 0;
    }

    switch (op) {
    case POWER:
      if (left == 0) {
        if (right < 0) {
          *error = Message{token.GetTokenProvenanceRange(opAt),
              "0 ** negative power"_err_en_US};
        }
      } else if (left != 1 && right != 1) {
        if (right <= 0) {
          left = !right;
        } else {
          std::int64_t power{1};
          for (; right > 0; --right) {
            if ((power * left) / left != power) {
              *error = Message{token.GetTokenProvenanceRange(opAt),
                  "overflow in exponentation"_err_en_US};
              left = 1;
            }
            power *= left;
          }
          left = power;
        }
      }
      break;
    case TIMES:
      if (left != 0 && right != 0 && ((left * right) / left) != right) {
        *error = Message{token.GetTokenProvenanceRange(opAt),
            "overflow in multiplication"_err_en_US};
      }
      left = left * right;
      break;
    case DIVIDE:
      if (right == 0) {
        *error = Message{
            token.GetTokenProvenanceRange(opAt), "division by zero"_err_en_US};
        left = 0;
      } else {
        left = left / right;
      }
      break;
    case MODULUS:
      if (right == 0) {
        *error = Message{
            token.GetTokenProvenanceRange(opAt), "modulus by zero"_err_en_US};
        left = 0;
      } else {
        left = left % right;
      }
      break;
    case ADD:
      if ((left < 0) == (right < 0) && (left < 0) != (left + right < 0)) {
        *error = Message{token.GetTokenProvenanceRange(opAt),
            "overflow in addition"_err_en_US};
      }
      left = left + right;
      break;
    case SUBTRACT:
      if ((left < 0) != (right < 0) && (left < 0) == (left - right < 0)) {
        *error = Message{token.GetTokenProvenanceRange(opAt),
            "overflow in subtraction"_err_en_US};
      }
      left = left - right;
      break;
    case LEFTSHIFT:
      if (right < 0 || right > 64) {
        *error = Message{token.GetTokenProvenanceRange(opAt),
            "bad left shift count"_err_en_US};
      }
      left = right >= 64 ? 0 : left << right;
      break;
    case RIGHTSHIFT:
      if (right < 0 || right > 64) {
        *error = Message{token.GetTokenProvenanceRange(opAt),
            "bad right shift count"_err_en_US};
      }
      left = right >= 64 ? 0 : left >> right;
      break;
    case BITAND:
    case AND:
      left = left & right;
      break;
    case BITXOR:
      left = left ^ right;
      break;
    case BITOR:
    case OR:
      left = left | right;
      break;
    case LT:
      left = -(left < right);
      break;
    case LE:
      left = -(left <= right);
      break;
    case EQ:
      left = -(left == right);
      break;
    case NE:
      left = -(left != right);
      break;
    case GE:
      left = -(left >= right);
      break;
    case GT:
      left = -(left > right);
      break;
    case EQV:
      left = -(!left == !right);
      break;
    case NEQV:
      left = -(!left != !right);
      break;
    case SELECT:
      if (*atToken >= tokens || token.TokenAt(*atToken).ToString() != ":") {
        *error = Message{token.GetTokenProvenanceRange(opAt),
            "':' required in selection expression"_err_en_US};
        return 0;
      } else {
        ++*atToken;
        std::int64_t third{
            ExpressionValue(token, operandPrecedence[op], atToken, error)};
        left = left != 0 ? right : third;
      }
      break;
    case COMMA:
      left = right;
      break;
    default:
      CRASH_NO_CASE;
    }
  }
  return left;
}

bool Preprocessor::IsIfPredicateTrue(const TokenSequence &expr,
    std::size_t first, std::size_t exprTokens, Prescanner &prescanner) {
  TokenSequence expr1{expr, first, exprTokens};
  if (expr1.HasBlanks()) {
    expr1.RemoveBlanks();
  }
  TokenSequence expr2;
  for (std::size_t j{0}; j < expr1.SizeInTokens(); ++j) {
    if (ToLowerCaseLetters(expr1.TokenAt(j).ToString()) == "defined") {
      CharBlock name;
      if (j + 3 < expr1.SizeInTokens() &&
          expr1.TokenAt(j + 1).ToString() == "(" &&
          expr1.TokenAt(j + 3).ToString() == ")") {
        name = expr1.TokenAt(j + 2);
        j += 3;
      } else if (j + 1 < expr1.SizeInTokens() &&
          IsLegalIdentifierStart(expr1.TokenAt(j + 1))) {
        name = expr1.TokenAt(++j);
      }
      if (!name.empty()) {
        char truth{IsNameDefined(name) ? '1' : '0'};
        expr2.Put(&truth, 1, allSources_.CompilerInsertionProvenance(truth));
        continue;
      }
    }
    expr2.Put(expr1, j);
  }
  TokenSequence expr3{ReplaceMacros(expr2, prescanner)};
  if (expr3.HasBlanks()) {
    expr3.RemoveBlanks();
  }
  if (expr3.empty()) {
    prescanner.Say(expr.GetProvenanceRange(), "empty expression"_err_en_US);
    return false;
  }
  std::size_t atToken{0};
  std::optional<Message> error;
  bool result{ExpressionValue(expr3, 0, &atToken, &error) != 0};
  if (error) {
    prescanner.Say(std::move(*error));
  } else if (atToken < expr3.SizeInTokens() &&
      expr3.TokenAt(atToken).ToString() != "!") {
    prescanner.Say(expr3.GetIntervalProvenanceRange(
                       atToken, expr3.SizeInTokens() - atToken),
        atToken == 0 ? "could not parse any expression"_err_en_US
                     : "excess characters after expression"_err_en_US);
  }
  return result;
}
} // namespace Fortran::parser
