//===--- ASTConsumers.h - ASTConsumer implementations -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Consumers.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVER_ASTCONSUMERS_H
#define DRIVER_ASTCONSUMERS_H

#include "clang/Basic/LLVM.h"

namespace llvm {
  namespace sys { class Path; }
}
namespace clang {

class ASTConsumer;
class CodeGenOptions;
class DiagnosticsEngine;
class FileManager;
class LangOptions;
class Preprocessor;
class TargetOptions;

// AST pretty-printer: prints out the AST in a format that is close to the
// original C code.  The output is intended to be in a format such that
// clang could re-parse the output back into the same AST, but the
// implementation is still incomplete.
ASTConsumer *CreateASTPrinter(raw_ostream *OS, StringRef FilterString);

// AST dumper: dumps the raw AST in human-readable form to stderr; this is
// intended for debugging.
ASTConsumer *CreateASTDumper(StringRef FilterString);

// AST Decl node lister: prints qualified names of all filterable AST Decl
// nodes.
ASTConsumer *CreateASTDeclNodeLister();

// AST XML-dumper: dumps out the AST to stderr in a very detailed XML
// format; this is intended for particularly intense debugging.
ASTConsumer *CreateASTDumperXML(raw_ostream &OS);

// Graphical AST viewer: for each function definition, creates a graph of
// the AST and displays it with the graph viewer "dotty".  Also outputs
// function declarations to stderr.
ASTConsumer *CreateASTViewer();

// DeclContext printer: prints out the DeclContext tree in human-readable form
// to stderr; this is intended for debugging.
ASTConsumer *CreateDeclContextPrinter();

} // end clang namespace

#endif
