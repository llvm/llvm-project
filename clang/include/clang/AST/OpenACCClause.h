//===- OpenACCClause.h - Classes for OpenACC clauses ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file defines OpenACC AST classes for clauses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OPENACCCLAUSE_H
#define LLVM_CLANG_AST_OPENACCCLAUSE_H
#include "clang/AST/ASTContext.h"
#include "clang/Basic/OpenACCKinds.h"

namespace clang {
/// This is the base type for all OpenACC Clauses.
class OpenACCClause {
  OpenACCClauseKind Kind;
  SourceRange Location;

protected:
  OpenACCClause(OpenACCClauseKind K, SourceLocation BeginLoc,
                SourceLocation EndLoc)
      : Kind(K), Location(BeginLoc, EndLoc) {}

public:
  OpenACCClauseKind getClauseKind() const { return Kind; }
  SourceLocation getBeginLoc() const { return Location.getBegin(); }
  SourceLocation getEndLoc() const { return Location.getEnd(); }

  static bool classof(const OpenACCClause *) { return true; }

  virtual ~OpenACCClause() = default;
};

/// Represents a clause that has a list of parameters.
class OpenACCClauseWithParams : public OpenACCClause {
  /// Location of the '('.
  SourceLocation LParenLoc;

protected:
  OpenACCClauseWithParams(OpenACCClauseKind K, SourceLocation BeginLoc,
                          SourceLocation LParenLoc, SourceLocation EndLoc)
      : OpenACCClause(K, BeginLoc, EndLoc), LParenLoc(LParenLoc) {}

public:
  SourceLocation getLParenLoc() const { return LParenLoc; }
};

template <class Impl> class OpenACCClauseVisitor {
  Impl &getDerived() { return static_cast<Impl &>(*this); }

public:
  void VisitClauseList(ArrayRef<const OpenACCClause *> List) {
    for (const OpenACCClause *Clause : List)
      Visit(Clause);
  }

  void Visit(const OpenACCClause *C) {
    if (!C)
      return;

    switch (C->getClauseKind()) {
    case OpenACCClauseKind::Default:
    case OpenACCClauseKind::Finalize:
    case OpenACCClauseKind::IfPresent:
    case OpenACCClauseKind::Seq:
    case OpenACCClauseKind::Independent:
    case OpenACCClauseKind::Auto:
    case OpenACCClauseKind::Worker:
    case OpenACCClauseKind::Vector:
    case OpenACCClauseKind::NoHost:
    case OpenACCClauseKind::If:
    case OpenACCClauseKind::Self:
    case OpenACCClauseKind::Copy:
    case OpenACCClauseKind::UseDevice:
    case OpenACCClauseKind::Attach:
    case OpenACCClauseKind::Delete:
    case OpenACCClauseKind::Detach:
    case OpenACCClauseKind::Device:
    case OpenACCClauseKind::DevicePtr:
    case OpenACCClauseKind::DeviceResident:
    case OpenACCClauseKind::FirstPrivate:
    case OpenACCClauseKind::Host:
    case OpenACCClauseKind::Link:
    case OpenACCClauseKind::NoCreate:
    case OpenACCClauseKind::Present:
    case OpenACCClauseKind::Private:
    case OpenACCClauseKind::CopyOut:
    case OpenACCClauseKind::CopyIn:
    case OpenACCClauseKind::Create:
    case OpenACCClauseKind::Reduction:
    case OpenACCClauseKind::Collapse:
    case OpenACCClauseKind::Bind:
    case OpenACCClauseKind::VectorLength:
    case OpenACCClauseKind::NumGangs:
    case OpenACCClauseKind::NumWorkers:
    case OpenACCClauseKind::DeviceNum:
    case OpenACCClauseKind::DefaultAsync:
    case OpenACCClauseKind::DeviceType:
    case OpenACCClauseKind::DType:
    case OpenACCClauseKind::Async:
    case OpenACCClauseKind::Tile:
    case OpenACCClauseKind::Gang:
    case OpenACCClauseKind::Wait:
    case OpenACCClauseKind::Invalid:
      llvm_unreachable("Clause visitor not yet implemented");
    }
    llvm_unreachable("Invalid Clause kind");
  }
};

class OpenACCClausePrinter final
    : public OpenACCClauseVisitor<OpenACCClausePrinter> {
  raw_ostream &OS;

public:
  void VisitClauseList(ArrayRef<const OpenACCClause *> List) {
    for (const OpenACCClause *Clause : List) {
      Visit(Clause);

      if (Clause != List.back())
        OS << ' ';
    }
  }
  OpenACCClausePrinter(raw_ostream &OS) : OS(OS) {}
};

} // namespace clang

#endif // LLVM_CLANG_AST_OPENACCCLAUSE_H
