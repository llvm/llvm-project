//===-- lib/Parser/parsing.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/parsing.h"
#include "preprocessor.h"
#include "prescan.h"
#include "type-parsers.h"
#include "flang/Parser/message.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::parser {

Parsing::Parsing(AllCookedSources &allCooked) : allCooked_{allCooked} {}
Parsing::~Parsing() {}

const SourceFile *Parsing::Prescan(const std::string &path, Options options) {
  options_ = options;
  AllSources &allSources{allCooked_.allSources()};
  if (options.isModuleFile) {
    for (const auto &path : options.searchDirectories) {
      allSources.PushSearchPathDirectory(path);
    }
  }

  std::string buf;
  llvm::raw_string_ostream fileError{buf};
  const SourceFile *sourceFile;
  if (path == "-") {
    sourceFile = allSources.ReadStandardInput(fileError);
  } else {
    sourceFile = allSources.Open(path, fileError);
  }
  if (!fileError.str().empty()) {
    ProvenanceRange range{allSources.AddCompilerInsertion(path)};
    messages_.Say(range, "%s"_err_en_US, fileError.str());
    return sourceFile;
  }
  CHECK(sourceFile);

  if (!options.isModuleFile) {
    // For .mod files we always want to look in the search directories.
    // For normal source files we don't push them until after the primary
    // source file has been opened.  If foo.f is missing from the current
    // working directory, we don't want to accidentally read another foo.f
    // from another directory that's on the search path.
    for (const auto &path : options.searchDirectories) {
      allSources.PushSearchPathDirectory(path);
    }
  }

  Preprocessor preprocessor{allSources};
  for (const auto &predef : options.predefinitions) {
    if (predef.second) {
      preprocessor.Define(predef.first, *predef.second);
    } else {
      preprocessor.Undefine(predef.first);
    }
  }
  currentCooked_ = &allCooked_.NewCookedSource();
  Prescanner prescanner{
      messages_, *currentCooked_, preprocessor, options.features};
  prescanner.set_fixedForm(options.isFixedForm)
      .set_fixedFormColumnLimit(options.fixedFormColumns)
      .AddCompilerDirectiveSentinel("dir$");
  if (options.features.IsEnabled(LanguageFeature::OpenACC)) {
    prescanner.AddCompilerDirectiveSentinel("$acc");
  }
  if (options.features.IsEnabled(LanguageFeature::OpenMP)) {
    prescanner.AddCompilerDirectiveSentinel("$omp");
    prescanner.AddCompilerDirectiveSentinel("$"); // OMP conditional line
  }
  ProvenanceRange range{allSources.AddIncludedFile(
      *sourceFile, ProvenanceRange{}, options.isModuleFile)};
  prescanner.Prescan(range);
  if (currentCooked_->BufferedBytes() == 0 && !options.isModuleFile) {
    // Input is empty.  Append a newline so that any warning
    // message about nonstandard usage will have provenance.
    currentCooked_->Put('\n', range.start());
  }
  currentCooked_->Marshal(allSources);
  if (options.needProvenanceRangeToCharBlockMappings) {
    currentCooked_->CompileProvenanceRangeToOffsetMappings(allSources);
  }
  return sourceFile;
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
