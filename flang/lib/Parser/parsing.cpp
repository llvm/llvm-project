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

Parsing::Parsing(AllSources &s) : cooked_{s} {}
Parsing::~Parsing() {}

const SourceFile *Parsing::Prescan(const std::string &path, Options options) {
  options_ = options;
  AllSources &allSources{cooked_.allSources()};
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
  Prescanner prescanner{messages_, cooked_, preprocessor, options.features};
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
  if (cooked_.BufferedBytes() == 0 && !options.isModuleFile) {
    // Input is empty.  Append a newline so that any warning
    // message about nonstandard usage will have provenance.
    cooked_.Put('\n', range.start());
  }
  cooked_.Marshal();
  if (options.needProvenanceRangeToCharBlockMappings) {
    cooked_.CompileProvenanceRangeToOffsetMappings();
  }
  return sourceFile;
}

void Parsing::DumpCookedChars(llvm::raw_ostream &out) const {
  UserState userState{cooked_, common::LanguageFeatureControl{}};
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm).set_userState(&userState);
  while (std::optional<const char *> p{parseState.GetNextChar()}) {
    out << **p;
  }
}

void Parsing::DumpProvenance(llvm::raw_ostream &out) const {
  cooked_.Dump(out);
}

void Parsing::DumpParsingLog(llvm::raw_ostream &out) const {
  log_.Dump(out, cooked_);
}

void Parsing::Parse(llvm::raw_ostream &out) {
  UserState userState{cooked_, options_.features};
  userState.set_debugOutput(out)
      .set_instrumentedParse(options_.instrumentedParse)
      .set_log(&log_);
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm).set_userState(&userState);
  parseTree_ = program.Parse(parseState);
  CHECK(
      !parseState.anyErrorRecovery() || parseState.messages().AnyFatalError());
  consumedWholeFile_ = parseState.IsAtEnd();
  messages_.Annex(std::move(parseState.messages()));
  finalRestingPlace_ = parseState.GetLocation();
}

void Parsing::ClearLog() { log_.clear(); }

bool Parsing::ForTesting(std::string path, llvm::raw_ostream &err) {
  llvm::raw_null_ostream NullStream;
  Prescan(path, Options{});
  if (messages_.AnyFatalError()) {
    messages_.Emit(err, cooked_);
    err << "could not scan " << path << '\n';
    return false;
  }
  Parse(NullStream);
  messages_.Emit(err, cooked_);
  if (!consumedWholeFile_) {
    EmitMessage(err, finalRestingPlace_, "parser FAIL; final position");
    return false;
  }
  if (messages_.AnyFatalError() || !parseTree_.has_value()) {
    err << "could not parse " << path << '\n';
    return false;
  }
  return true;
}
} // namespace Fortran::parser
