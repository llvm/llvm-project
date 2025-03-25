//===- ParserActions.h -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_ACTIONS_H_
#define FORTRAN_PARSER_ACTIONS_H_

#include <string>

namespace llvm {
class raw_string_ostream;
class raw_ostream;
class StringRef;
} // namespace llvm

namespace Fortran::lower {
class LoweringBridge;
} // namespace Fortran::lower

namespace Fortran::parser {
class Parsing;
class AllCookedSources;
} // namespace Fortran::parser

namespace lower::pft {
class Program;
} // namespace lower::pft

//=== Frontend Parser helpers ===

namespace Fortran::frontend {
class CompilerInstance;

parser::AllCookedSources &getAllCooked(CompilerInstance &ci);

void parseAndLowerTree(CompilerInstance &ci, lower::LoweringBridge &lb);

void dumpTree(CompilerInstance &ci);

void dumpProvenance(CompilerInstance &ci);

void dumpPreFIRTree(CompilerInstance &ci);

void formatOrDumpPrescanner(std::string &buf,
                            llvm::raw_string_ostream &outForPP,
                            CompilerInstance &ci);

void debugMeasureParseTree(CompilerInstance &ci, llvm::StringRef filename);

void debugUnparseNoSema(CompilerInstance &ci, llvm::raw_ostream &out);

void debugUnparseWithSymbols(CompilerInstance &ci);

void debugUnparseWithModules(CompilerInstance &ci);

void debugDumpParsingLog(CompilerInstance &ci);
} // namespace Fortran::frontend

#endif // FORTRAN_PARSER_ACTIONS_H_
