//===- Nodes.cpp ----------------------------------------------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Basic/TokenKinds.h"

using namespace clang;

llvm::raw_ostream &syntax::operator<<(llvm::raw_ostream &OS, NodeKind K) {
  switch (K) {
  case NodeKind::Leaf:
    return OS << "Leaf";
  case NodeKind::TranslationUnit:
    return OS << "TranslationUnit";
  case NodeKind::UnknownExpression:
    return OS << "UnknownExpression";
  case NodeKind::UnknownStatement:
    return OS << "UnknownStatement";
  case NodeKind::DeclarationStatement:
    return OS << "DeclarationStatement";
  case NodeKind::EmptyStatement:
    return OS << "EmptyStatement";
  case NodeKind::SwitchStatement:
    return OS << "SwitchStatement";
  case NodeKind::CaseStatement:
    return OS << "CaseStatement";
  case NodeKind::DefaultStatement:
    return OS << "DefaultStatement";
  case NodeKind::IfStatement:
    return OS << "IfStatement";
  case NodeKind::ForStatement:
    return OS << "ForStatement";
  case NodeKind::WhileStatement:
    return OS << "WhileStatement";
  case NodeKind::ContinueStatement:
    return OS << "ContinueStatement";
  case NodeKind::BreakStatement:
    return OS << "BreakStatement";
  case NodeKind::ReturnStatement:
    return OS << "ReturnStatement";
  case NodeKind::RangeBasedForStatement:
    return OS << "RangeBasedForStatement";
  case NodeKind::ExpressionStatement:
    return OS << "ExpressionStatement";
  case NodeKind::CompoundStatement:
    return OS << "CompoundStatement";
  case NodeKind::UnknownDeclaration:
    return OS << "UnknownDeclaration";
  case NodeKind::EmptyDeclaration:
    return OS << "EmptyDeclaration";
  case NodeKind::StaticAssertDeclaration:
    return OS << "StaticAssertDeclaration";
  case NodeKind::LinkageSpecificationDeclaration:
    return OS << "LinkageSpecificationDeclaration";
  case NodeKind::SimpleDeclaration:
    return OS << "SimpleDeclaration";
  case NodeKind::NamespaceDefinition:
    return OS << "NamespaceDefinition";
  case NodeKind::NamespaceAliasDefinition:
    return OS << "NamespaceAliasDefinition";
  case NodeKind::UsingNamespaceDirective:
    return OS << "UsingNamespaceDirective";
  case NodeKind::UsingDeclaration:
    return OS << "UsingDeclaration";
  case NodeKind::TypeAliasDeclaration:
    return OS << "TypeAliasDeclaration";
  }
  llvm_unreachable("unknown node kind");
}

llvm::raw_ostream &syntax::operator<<(llvm::raw_ostream &OS, NodeRole R) {
  switch (R) {
  case syntax::NodeRole::Detached:
    return OS << "Detached";
  case syntax::NodeRole::Unknown:
    return OS << "Unknown";
  case syntax::NodeRole::OpenParen:
    return OS << "OpenParen";
  case syntax::NodeRole::CloseParen:
    return OS << "CloseParen";
  case syntax::NodeRole::IntroducerKeyword:
    return OS << "IntroducerKeyword";
  case syntax::NodeRole::BodyStatement:
    return OS << "BodyStatement";
  case syntax::NodeRole::CaseStatement_value:
    return OS << "CaseStatement_value";
  case syntax::NodeRole::IfStatement_thenStatement:
    return OS << "IfStatement_thenStatement";
  case syntax::NodeRole::IfStatement_elseKeyword:
    return OS << "IfStatement_elseKeyword";
  case syntax::NodeRole::IfStatement_elseStatement:
    return OS << "IfStatement_elseStatement";
  case syntax::NodeRole::ReturnStatement_value:
    return OS << "ReturnStatement_value";
  case syntax::NodeRole::ExpressionStatement_expression:
    return OS << "ExpressionStatement_expression";
  case syntax::NodeRole::CompoundStatement_statement:
    return OS << "CompoundStatement_statement";
  case syntax::NodeRole::StaticAssertDeclaration_condition:
    return OS << "StaticAssertDeclaration_condition";
  case syntax::NodeRole::StaticAssertDeclaration_message:
    return OS << "StaticAssertDeclaration_message";
  }
  llvm_unreachable("invalid role");
}

syntax::Leaf *syntax::SwitchStatement::switchKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::SwitchStatement::body() {
  return llvm::cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::CaseStatement::caseKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Expression *syntax::CaseStatement::value() {
  return llvm::cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::CaseStatement_value));
}

syntax::Statement *syntax::CaseStatement::body() {
  return llvm::cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::DefaultStatement::defaultKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::DefaultStatement::body() {
  return llvm::cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::IfStatement::ifKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::IfStatement::thenStatement() {
  return llvm::cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::IfStatement_thenStatement));
}

syntax::Leaf *syntax::IfStatement::elseKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IfStatement_elseKeyword));
}

syntax::Statement *syntax::IfStatement::elseStatement() {
  return llvm::cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::IfStatement_elseStatement));
}

syntax::Leaf *syntax::ForStatement::forKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::ForStatement::body() {
  return llvm::cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::WhileStatement::whileKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::WhileStatement::body() {
  return llvm::cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::ContinueStatement::continueKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::BreakStatement::breakKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::ReturnStatement::returnKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Expression *syntax::ReturnStatement::value() {
  return llvm::cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::ReturnStatement_value));
}

syntax::Leaf *syntax::RangeBasedForStatement::forKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::RangeBasedForStatement::body() {
  return llvm::cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Expression *syntax::ExpressionStatement::expression() {
  return llvm::cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::ExpressionStatement_expression));
}

syntax::Leaf *syntax::CompoundStatement::lbrace() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::OpenParen));
}

std::vector<syntax::Statement *> syntax::CompoundStatement::statements() {
  std::vector<syntax::Statement *> Children;
  for (auto *C = firstChild(); C; C = C->nextSibling()) {
    if (C->role() == syntax::NodeRole::CompoundStatement_statement)
      Children.push_back(llvm::cast<syntax::Statement>(C));
  }
  return Children;
}

syntax::Leaf *syntax::CompoundStatement::rbrace() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::CloseParen));
}

syntax::Expression *syntax::StaticAssertDeclaration::condition() {
  return llvm::cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::StaticAssertDeclaration_condition));
}

syntax::Expression *syntax::StaticAssertDeclaration::message() {
  return llvm::cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::StaticAssertDeclaration_message));
}
