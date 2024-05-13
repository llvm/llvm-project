//===-- examples/flang-omp-report-plugin/flang-omp-report.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This plugin parses a Fortran source file and generates a YAML report with
// all the OpenMP constructs and clauses and which line they're located on.
//
// The plugin may be invoked as:
// ./bin/flang-new -fc1 -load lib/flangOmpReport.so -plugin flang-omp-report
// -fopenmp
//
//===----------------------------------------------------------------------===//

#include "FlangOmpReportVisitor.h"

#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendPluginRegistry.h"
#include "flang/Parser/dump-parse-tree.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

using namespace Fortran::frontend;
using namespace Fortran::parser;

LLVM_YAML_IS_SEQUENCE_VECTOR(LogRecord)
LLVM_YAML_IS_SEQUENCE_VECTOR(ClauseInfo)
namespace llvm {
namespace yaml {
using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
template <> struct MappingTraits<ClauseInfo> {
  static void mapping(IO &io, ClauseInfo &info) {
    io.mapRequired("clause", info.clause);
    io.mapRequired("details", info.clauseDetails);
  }
};
template <> struct MappingTraits<LogRecord> {
  static void mapping(IO &io, LogRecord &info) {
    io.mapRequired("file", info.file);
    io.mapRequired("line", info.line);
    io.mapRequired("construct", info.construct);
    io.mapRequired("clauses", info.clauses);
  }
};
} // namespace yaml
} // namespace llvm

class FlangOmpReport : public PluginParseTreeAction {
  void executeAction() override {
    // Prepare the parse tree and the visitor
    Parsing &parsing = getParsing();
    OpenMPCounterVisitor visitor;
    visitor.parsing = &parsing;

    // Walk the parse tree
    Walk(parsing.parseTree(), visitor);

    // Dump the output
    std::unique_ptr<llvm::raw_pwrite_stream> OS{
        createOutputFile(/*extension=*/"yaml")};
    llvm::yaml::Output yout(*OS);

    yout << visitor.constructClauses;
  }
};

static FrontendPluginRegistry::Add<FlangOmpReport> X("flang-omp-report",
    "Generate a YAML summary of OpenMP constructs and clauses");
