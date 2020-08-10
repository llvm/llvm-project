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

raw_ostream &syntax::operator<<(raw_ostream &OS, NodeKind K) {
  switch (K) {
  case NodeKind::Leaf:
    return OS << "Leaf";
  case NodeKind::TranslationUnit:
    return OS << "TranslationUnit";
  case NodeKind::UnknownExpression:
    return OS << "UnknownExpression";
  case NodeKind::ParenExpression:
    return OS << "ParenExpression";
  case NodeKind::IntegerLiteralExpression:
    return OS << "IntegerLiteralExpression";
  case NodeKind::CharacterLiteralExpression:
    return OS << "CharacterLiteralExpression";
  case NodeKind::FloatingLiteralExpression:
    return OS << "FloatingLiteralExpression";
  case NodeKind::StringLiteralExpression:
    return OS << "StringLiteralExpression";
  case NodeKind::BoolLiteralExpression:
    return OS << "BoolLiteralExpression";
  case NodeKind::CxxNullPtrExpression:
    return OS << "CxxNullPtrExpression";
  case NodeKind::IntegerUserDefinedLiteralExpression:
    return OS << "IntegerUserDefinedLiteralExpression";
  case NodeKind::FloatUserDefinedLiteralExpression:
    return OS << "FloatUserDefinedLiteralExpression";
  case NodeKind::CharUserDefinedLiteralExpression:
    return OS << "CharUserDefinedLiteralExpression";
  case NodeKind::StringUserDefinedLiteralExpression:
    return OS << "StringUserDefinedLiteralExpression";
  case NodeKind::PrefixUnaryOperatorExpression:
    return OS << "PrefixUnaryOperatorExpression";
  case NodeKind::PostfixUnaryOperatorExpression:
    return OS << "PostfixUnaryOperatorExpression";
  case NodeKind::BinaryOperatorExpression:
    return OS << "BinaryOperatorExpression";
  case NodeKind::UnqualifiedId:
    return OS << "UnqualifiedId";
  case NodeKind::IdExpression:
    return OS << "IdExpression";
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
  case NodeKind::TemplateDeclaration:
    return OS << "TemplateDeclaration";
  case NodeKind::ExplicitTemplateInstantiation:
    return OS << "ExplicitTemplateInstantiation";
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
  case NodeKind::SimpleDeclarator:
    return OS << "SimpleDeclarator";
  case NodeKind::ParenDeclarator:
    return OS << "ParenDeclarator";
  case NodeKind::ArraySubscript:
    return OS << "ArraySubscript";
  case NodeKind::TrailingReturnType:
    return OS << "TrailingReturnType";
  case NodeKind::ParametersAndQualifiers:
    return OS << "ParametersAndQualifiers";
  case NodeKind::MemberPointer:
    return OS << "MemberPointer";
  case NodeKind::GlobalNameSpecifier:
    return OS << "GlobalNameSpecifier";
  case NodeKind::DecltypeNameSpecifier:
    return OS << "DecltypeNameSpecifier";
  case NodeKind::IdentifierNameSpecifier:
    return OS << "IdentifierNameSpecifier";
  case NodeKind::SimpleTemplateNameSpecifier:
    return OS << "SimpleTemplateNameSpecifier";
  case NodeKind::NestedNameSpecifier:
    return OS << "NestedNameSpecifier";
  }
  llvm_unreachable("unknown node kind");
}

raw_ostream &syntax::operator<<(raw_ostream &OS, NodeRole R) {
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
  case syntax::NodeRole::LiteralToken:
    return OS << "LiteralToken";
  case syntax::NodeRole::ArrowToken:
    return OS << "ArrowToken";
  case syntax::NodeRole::ExternKeyword:
    return OS << "ExternKeyword";
  case syntax::NodeRole::TemplateKeyword:
    return OS << "TemplateKeyword";
  case syntax::NodeRole::BodyStatement:
    return OS << "BodyStatement";
  case syntax::NodeRole::List_element:
    return OS << "List_element";
  case syntax::NodeRole::List_delimiter:
    return OS << "List_delimiter";
  case syntax::NodeRole::CaseStatement_value:
    return OS << "CaseStatement_value";
  case syntax::NodeRole::IfStatement_thenStatement:
    return OS << "IfStatement_thenStatement";
  case syntax::NodeRole::IfStatement_elseKeyword:
    return OS << "IfStatement_elseKeyword";
  case syntax::NodeRole::IfStatement_elseStatement:
    return OS << "IfStatement_elseStatement";
  case syntax::NodeRole::OperatorExpression_operatorToken:
    return OS << "OperatorExpression_operatorToken";
  case syntax::NodeRole::UnaryOperatorExpression_operand:
    return OS << "UnaryOperatorExpression_operand";
  case syntax::NodeRole::BinaryOperatorExpression_leftHandSide:
    return OS << "BinaryOperatorExpression_leftHandSide";
  case syntax::NodeRole::BinaryOperatorExpression_rightHandSide:
    return OS << "BinaryOperatorExpression_rightHandSide";
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
  case syntax::NodeRole::SimpleDeclaration_declarator:
    return OS << "SimpleDeclaration_declarator";
  case syntax::NodeRole::TemplateDeclaration_declaration:
    return OS << "TemplateDeclaration_declaration";
  case syntax::NodeRole::ExplicitTemplateInstantiation_declaration:
    return OS << "ExplicitTemplateInstantiation_declaration";
  case syntax::NodeRole::ArraySubscript_sizeExpression:
    return OS << "ArraySubscript_sizeExpression";
  case syntax::NodeRole::TrailingReturnType_declarator:
    return OS << "TrailingReturnType_declarator";
  case syntax::NodeRole::ParametersAndQualifiers_parameter:
    return OS << "ParametersAndQualifiers_parameter";
  case syntax::NodeRole::ParametersAndQualifiers_trailingReturn:
    return OS << "ParametersAndQualifiers_trailingReturn";
  case syntax::NodeRole::IdExpression_id:
    return OS << "IdExpression_id";
  case syntax::NodeRole::IdExpression_qualifier:
    return OS << "IdExpression_qualifier";
  case syntax::NodeRole::ParenExpression_subExpression:
    return OS << "ParenExpression_subExpression";
  }
  llvm_unreachable("invalid role");
}

// We could have an interator in list to not pay memory costs of temporary
// vector
std::vector<syntax::NameSpecifier *> syntax::NestedNameSpecifier::specifiers() {
  auto specifiersAsNodes = getElementsAsNodes();
  std::vector<syntax::NameSpecifier *> Children;
  for (const auto &element : specifiersAsNodes) {
    Children.push_back(llvm::cast<syntax::NameSpecifier>(element));
  }
  return Children;
}

std::vector<syntax::List::ElementAndDelimiter<syntax::NameSpecifier>>
syntax::NestedNameSpecifier::specifiersAndDoubleColons() {
  auto specifiersAsNodesAndDoubleColons = getElementsAsNodesAndDelimiters();
  std::vector<syntax::List::ElementAndDelimiter<syntax::NameSpecifier>>
      Children;
  for (const auto &specifierAndDoubleColon : specifiersAsNodesAndDoubleColons) {
    Children.push_back(
        {llvm::cast<syntax::NameSpecifier>(specifierAndDoubleColon.element),
         specifierAndDoubleColon.delimiter});
  }
  return Children;
}

syntax::NestedNameSpecifier *syntax::IdExpression::qualifier() {
  return cast_or_null<syntax::NestedNameSpecifier>(
      findChild(syntax::NodeRole::IdExpression_qualifier));
}

syntax::Leaf *syntax::IdExpression::templateKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::TemplateKeyword));
}

syntax::UnqualifiedId *syntax::IdExpression::unqualifiedId() {
  return cast_or_null<syntax::UnqualifiedId>(
      findChild(syntax::NodeRole::IdExpression_id));
}

syntax::Leaf *syntax::ParenExpression::openParen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

syntax::Expression *syntax::ParenExpression::subExpression() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::ParenExpression_subExpression));
}

syntax::Leaf *syntax::ParenExpression::closeParen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Leaf *syntax::LiteralExpression::literalToken() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::LiteralToken));
}

syntax::Expression *syntax::BinaryOperatorExpression::lhs() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::BinaryOperatorExpression_leftHandSide));
}

syntax::Leaf *syntax::UnaryOperatorExpression::operatorToken() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::OperatorExpression_operatorToken));
}

syntax::Expression *syntax::UnaryOperatorExpression::operand() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::UnaryOperatorExpression_operand));
}

syntax::Leaf *syntax::BinaryOperatorExpression::operatorToken() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::OperatorExpression_operatorToken));
}

syntax::Expression *syntax::BinaryOperatorExpression::rhs() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::BinaryOperatorExpression_rightHandSide));
}

syntax::Leaf *syntax::SwitchStatement::switchKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::SwitchStatement::body() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::CaseStatement::caseKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Expression *syntax::CaseStatement::value() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::CaseStatement_value));
}

syntax::Statement *syntax::CaseStatement::body() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::DefaultStatement::defaultKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::DefaultStatement::body() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::IfStatement::ifKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::IfStatement::thenStatement() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::IfStatement_thenStatement));
}

syntax::Leaf *syntax::IfStatement::elseKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IfStatement_elseKeyword));
}

syntax::Statement *syntax::IfStatement::elseStatement() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::IfStatement_elseStatement));
}

syntax::Leaf *syntax::ForStatement::forKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::ForStatement::body() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::WhileStatement::whileKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::WhileStatement::body() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::ContinueStatement::continueKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::BreakStatement::breakKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::ReturnStatement::returnKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Expression *syntax::ReturnStatement::value() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::ReturnStatement_value));
}

syntax::Leaf *syntax::RangeBasedForStatement::forKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::RangeBasedForStatement::body() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Expression *syntax::ExpressionStatement::expression() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::ExpressionStatement_expression));
}

syntax::Leaf *syntax::CompoundStatement::lbrace() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

std::vector<syntax::Statement *> syntax::CompoundStatement::statements() {
  std::vector<syntax::Statement *> Children;
  for (auto *C = firstChild(); C; C = C->nextSibling()) {
    assert(C->role() == syntax::NodeRole::CompoundStatement_statement);
    Children.push_back(cast<syntax::Statement>(C));
  }
  return Children;
}

syntax::Leaf *syntax::CompoundStatement::rbrace() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Expression *syntax::StaticAssertDeclaration::condition() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::StaticAssertDeclaration_condition));
}

syntax::Expression *syntax::StaticAssertDeclaration::message() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::StaticAssertDeclaration_message));
}

std::vector<syntax::SimpleDeclarator *>
syntax::SimpleDeclaration::declarators() {
  std::vector<syntax::SimpleDeclarator *> Children;
  for (auto *C = firstChild(); C; C = C->nextSibling()) {
    if (C->role() == syntax::NodeRole::SimpleDeclaration_declarator)
      Children.push_back(cast<syntax::SimpleDeclarator>(C));
  }
  return Children;
}

syntax::Leaf *syntax::TemplateDeclaration::templateKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Declaration *syntax::TemplateDeclaration::declaration() {
  return cast_or_null<syntax::Declaration>(
      findChild(syntax::NodeRole::TemplateDeclaration_declaration));
}

syntax::Leaf *syntax::ExplicitTemplateInstantiation::templateKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::ExplicitTemplateInstantiation::externKeyword() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::ExternKeyword));
}

syntax::Declaration *syntax::ExplicitTemplateInstantiation::declaration() {
  return cast_or_null<syntax::Declaration>(
      findChild(syntax::NodeRole::ExplicitTemplateInstantiation_declaration));
}

syntax::Leaf *syntax::ParenDeclarator::lparen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

syntax::Leaf *syntax::ParenDeclarator::rparen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Leaf *syntax::ArraySubscript::lbracket() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

syntax::Expression *syntax::ArraySubscript::sizeExpression() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::ArraySubscript_sizeExpression));
}

syntax::Leaf *syntax::ArraySubscript::rbracket() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Leaf *syntax::TrailingReturnType::arrowToken() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::ArrowToken));
}

syntax::SimpleDeclarator *syntax::TrailingReturnType::declarator() {
  return cast_or_null<syntax::SimpleDeclarator>(
      findChild(syntax::NodeRole::TrailingReturnType_declarator));
}

syntax::Leaf *syntax::ParametersAndQualifiers::lparen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

std::vector<syntax::SimpleDeclaration *>
syntax::ParametersAndQualifiers::parameters() {
  std::vector<syntax::SimpleDeclaration *> Children;
  for (auto *C = firstChild(); C; C = C->nextSibling()) {
    if (C->role() == syntax::NodeRole::ParametersAndQualifiers_parameter)
      Children.push_back(cast<syntax::SimpleDeclaration>(C));
  }
  return Children;
}

syntax::Leaf *syntax::ParametersAndQualifiers::rparen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::TrailingReturnType *syntax::ParametersAndQualifiers::trailingReturn() {
  return cast_or_null<syntax::TrailingReturnType>(
      findChild(syntax::NodeRole::ParametersAndQualifiers_trailingReturn));
}
