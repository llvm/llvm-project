//===-- lib/Parser/parsing.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/parsing.h"
#include "prescan.h"
#include "type-parsers.h"
#include "flang/Parser/message.h"
#include "flang/Parser/preprocessor.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::parser {

Parsing::Parsing(AllCookedSources &allCooked) : allCooked_{allCooked} {}
Parsing::~Parsing() {}

const SourceFile *Parsing::Prescan(const std::string &path, Options options) {
  options_ = options;
  AllSources &allSources{allCooked_.allSources()};
  allSources.ClearSearchPath();
  if (options.isModuleFile) {
    for (const auto &path : options.searchDirectories) {
      allSources.AppendSearchPathDirectory(path);
    }
  }

  std::string buf;
  llvm::raw_string_ostream fileError{buf};
  const SourceFile *sourceFile{nullptr};
  if (path == "-") {
    sourceFile = allSources.ReadStandardInput(fileError);
  } else if (options.isModuleFile) {
    // Don't mess with intrinsic module search path
    sourceFile = allSources.Open(path, fileError);
  } else {
    sourceFile =
        allSources.Open(path, fileError, "."s /*prepend to search path*/);
  }
  if (!buf.empty()) {
    ProvenanceRange range{allSources.AddCompilerInsertion(path)};
    messages_.Say(range, "%s"_err_en_US, buf);
    return sourceFile;
  }
  CHECK(sourceFile);

  if (!options.isModuleFile) {
    // For .mod files we always want to look in the search directories.
    // For normal source files we don't add them until after the primary
    // source file has been opened.  If foo.f is missing from the current
    // working directory, we don't want to accidentally read another foo.f
    // from another directory that's on the search path.
    for (const auto &path : options.searchDirectories) {
      allSources.AppendSearchPathDirectory(path);
    }
  }

  if (!options.predefinitions.empty()) {
    preprocessor_.DefineStandardMacros();
    for (const auto &predef : options.predefinitions) {
      if (predef.second) {
        preprocessor_.Define(predef.first, *predef.second);
      } else {
        preprocessor_.Undefine(predef.first);
      }
    }
  }
  currentCooked_ = &allCooked_.NewCookedSource();
  Prescanner prescanner{
      messages_, *currentCooked_, preprocessor_, options.features};
  prescanner.set_fixedForm(options.isFixedForm)
      .set_fixedFormColumnLimit(options.fixedFormColumns)
      .set_expandIncludeLines(!options.prescanAndReformat ||
          options.expandIncludeLinesInPreprocessedOutput)
      .AddCompilerDirectiveSentinel("dir$");
  if (options.features.IsEnabled(LanguageFeature::OpenACC)) {
    prescanner.AddCompilerDirectiveSentinel("$acc");
  }
  if (options.features.IsEnabled(LanguageFeature::OpenMP)) {
    prescanner.AddCompilerDirectiveSentinel("$omp");
    prescanner.AddCompilerDirectiveSentinel("$"); // OMP conditional line
  }
  if (options.features.IsEnabled(LanguageFeature::CUDA)) {
    prescanner.AddCompilerDirectiveSentinel("$cuf");
    prescanner.AddCompilerDirectiveSentinel("@cuf");
    preprocessor_.Define("_CUDA", "1");
  }
  ProvenanceRange range{allSources.AddIncludedFile(
      *sourceFile, ProvenanceRange{}, options.isModuleFile)};
  prescanner.Prescan(range);
  if (currentCooked_->BufferedBytes() == 0 && !options.isModuleFile) {
    // Input is empty.  Append a newline so that any warning
    // message about nonstandard usage will have provenance.
    currentCooked_->Put('\n', range.start());
  }
  currentCooked_->Marshal(allCooked_);
  if (options.needProvenanceRangeToCharBlockMappings) {
    currentCooked_->CompileProvenanceRangeToOffsetMappings(allSources);
  }
  if (options.showColors) {
    allSources.setShowColors(/*showColors=*/true);
  }
  return sourceFile;
}

void Parsing::EmitPreprocessorMacros(llvm::raw_ostream &out) const {
  preprocessor_.PrintMacros(out);
}

void Parsing::EmitPreprocessedSource(
    llvm::raw_ostream &out, bool lineDirectives) const {
  const std::string *sourcePath{nullptr};
  int sourceLine{0};
  int column{1};
  bool inDirective{false};
  bool inContinuation{false};
  bool lineWasBlankBefore{true};
  const AllSources &allSources{allCooked().allSources()};
  // All directives that flang support are known to have a length of 3 chars
  constexpr int directiveNameLength{3};
  // We need to know the current directive in order to provide correct
  // continuation for the directive
  std::string directive;
  for (const char &atChar : cooked().AsCharBlock()) {
    char ch{atChar};
    if (ch == '\n') {
      out << '\n'; // TODO: DOS CR-LF line ending if necessary
      column = 1;
      inDirective = false;
      inContinuation = false;
      lineWasBlankBefore = true;
      ++sourceLine;
      directive.clear();
    } else {
      auto provenance{cooked().GetProvenanceRange(CharBlock{&atChar, 1})};

      // Preserves original case of the character
      const auto getOriginalChar{[&](char ch) {
        if (IsLetter(ch) && provenance && provenance->size() == 1) {
          if (const char *orig{allSources.GetSource(*provenance)}) {
            const char upper{ToUpperCaseLetter(ch)};
            if (*orig == upper) {
              return upper;
            }
          }
        }
        return ch;
      }};

      if (ch == '!' && lineWasBlankBefore) {
        // Other comment markers (C, *, D) in original fixed form source
        // input card column 1 will have been deleted or normalized to !,
        // which signifies a comment (directive) in both source forms.
        inDirective = true;
      }
      if (inDirective && directive.size() < directiveNameLength &&
          IsLetter(ch)) {
        directive += getOriginalChar(ch);
      }

      std::optional<SourcePosition> position{provenance
              ? allSources.GetSourcePosition(provenance->start())
              : std::nullopt};
      if (lineDirectives && column == 1 && position) {
        if (&*position->path != sourcePath) {
          out << "#line \"" << *position->path << "\" " << position->line
              << '\n';
        } else if (position->line != sourceLine) {
          if (sourceLine < position->line &&
              sourceLine + 10 >= position->line) {
            // Emit a few newlines to catch up when they'll likely
            // require fewer bytes than a #line directive would have
            // occupied.
            while (sourceLine++ < position->line) {
              out << '\n';
            }
          } else {
            out << "#line " << position->line << '\n';
          }
        }
        sourcePath = &*position->path;
        sourceLine = position->line;
      }
      if (column > 72) {
        // Wrap long lines in a portable fashion that works in both
        // of the Fortran source forms. The first free-form continuation
        // marker ("&") lands in column 73, which begins the card commentary
        // field of fixed form, and the second one is put in column 6,
        // where it signifies fixed form line continuation.
        // The standard Fortran fixed form column limit (72) is used
        // for output, even if the input was parsed with a nonstandard
        // column limit override option.
        // OpenMP and OpenACC directives' continuations should have the
        // corresponding sentinel at the next line.
        const auto continuation{
            inDirective ? "&\n!$" + directive + "&" : "&\n     &"s};
        out << continuation;
        column = 7; // start of fixed form source field
        ++sourceLine;
        inContinuation = true;
      } else if (!inDirective && ch != ' ' && (ch < '0' || ch > '9')) {
        // Put anything other than a label or directive into the
        // Fortran fixed form source field (columns [7:72]).
        for (; column < 7; ++column) {
          out << ' ';
        }
      }
      if (!inContinuation && position && position->column <= 72 && ch != ' ') {
        // Preserve original indentation
        for (; column < position->column; ++column) {
          out << ' ';
        }
      }
      out << getOriginalChar(ch);
      lineWasBlankBefore = ch == ' ' && lineWasBlankBefore;
      ++column;
    }
  }
}

void Parsing::DumpCookedChars(llvm::raw_ostream &out) const {
  UserState userState{allCooked_, common::LanguageFeatureControl{}};
  ParseState parseState{cooked()};
  parseState.set_inFixedForm(options_.isFixedForm).set_userState(&userState);
  while (std::optional<const char *> p{parseState.GetNextChar()}) {
    out << **p;
  }
}

void Parsing::DumpProvenance(llvm::raw_ostream &out) const {
  allCooked_.Dump(out);
}

void Parsing::DumpParsingLog(llvm::raw_ostream &out) const {
  log_.Dump(out, allCooked_);
}

void Parsing::Parse(llvm::raw_ostream &out) {
  UserState userState{allCooked_, options_.features};
  userState.set_debugOutput(out)
      .set_instrumentedParse(options_.instrumentedParse)
      .set_log(&log_);
  ParseState parseState{cooked()};
  parseState.set_inFixedForm(options_.isFixedForm).set_userState(&userState);
  parseTree_ = program.Parse(parseState);
  CHECK(
      !parseState.anyErrorRecovery() || parseState.messages().AnyFatalError());
  consumedWholeFile_ = parseState.IsAtEnd();
  messages_.Annex(std::move(parseState.messages()));
  finalRestingPlace_ = parseState.GetLocation();
}

void Parsing::ClearLog() { log_.clear(); }

} // namespace Fortran::parser
