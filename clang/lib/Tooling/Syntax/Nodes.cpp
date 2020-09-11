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
  case NodeKind::ThisExpression:
    return OS << "ThisExpression";
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
  case NodeKind::CallExpression:
    return OS << "CallExpression";
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
  case NodeKind::MemberExpression:
    return OS << "MemberExpression";
  case NodeKind::CallArguments:
    return OS << "CallArguments";
  case NodeKind::ParameterDeclarationList:
    return OS << "ParameterDeclarationList";
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
  case syntax::NodeRole::ListElement:
    return OS << "ListElement";
  case syntax::NodeRole::ListDelimiter:
    return OS << "ListDelimiter";
  case syntax::NodeRole::CaseValue:
    return OS << "CaseValue";
  case syntax::NodeRole::ReturnValue:
    return OS << "ReturnValue";
  case syntax::NodeRole::ThenStatement:
    return OS << "ThenStatement";
  case syntax::NodeRole::ElseKeyword:
    return OS << "ElseKeyword";
  case syntax::NodeRole::ElseStatement:
    return OS << "ElseStatement";
  case syntax::NodeRole::OperatorToken:
    return OS << "OperatorToken";
  case syntax::NodeRole::Operand:
    return OS << "Operand";
  case syntax::NodeRole::LeftHandSide:
    return OS << "LeftHandSide";
  case syntax::NodeRole::RightHandSide:
    return OS << "RightHandSide";
  case syntax::NodeRole::Expression:
    return OS << "Expression";
  case syntax::NodeRole::Statement:
    return OS << "Statement";
  case syntax::NodeRole::Condition:
    return OS << "Condition";
  case syntax::NodeRole::Message:
    return OS << "Message";
  case syntax::NodeRole::Declarator:
    return OS << "Declarator";
  case syntax::NodeRole::Declaration:
    return OS << "Declaration";
  case syntax::NodeRole::Size:
    return OS << "Size";
  case syntax::NodeRole::Parameters:
    return OS << "Parameters";
  case syntax::NodeRole::TrailingReturn:
    return OS << "TrailingReturn";
  case syntax::NodeRole::UnqualifiedId:
    return OS << "UnqualifiedId";
  case syntax::NodeRole::Qualifier:
    return OS << "Qualifier";
  case syntax::NodeRole::SubExpression:
    return OS << "SubExpression";
  case syntax::NodeRole::Object:
    return OS << "Object";
  case syntax::NodeRole::AccessToken:
    return OS << "AccessToken";
  case syntax::NodeRole::Member:
    return OS << "Member";
  case syntax::NodeRole::Callee:
    return OS << "Callee";
  case syntax::NodeRole::Arguments:
    return OS << "Arguments";
  }
  llvm_unreachable("invalid role");
}

// We could have an interator in list to not pay memory costs of temporary
// vector
std::vector<syntax::NameSpecifier *>
syntax::NestedNameSpecifier::getSpecifiers() {
  auto specifiersAsNodes = getElementsAsNodes();
  std::vector<syntax::NameSpecifier *> Children;
  for (const auto &element : specifiersAsNodes) {
    Children.push_back(llvm::cast<syntax::NameSpecifier>(element));
  }
  return Children;
}

std::vector<syntax::List::ElementAndDelimiter<syntax::NameSpecifier>>
syntax::NestedNameSpecifier::getSpecifiersAndDoubleColons() {
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

std::vector<syntax::Expression *> syntax::CallArguments::getArguments() {
  auto ArgumentsAsNodes = getElementsAsNodes();
  std::vector<syntax::Expression *> Children;
  for (const auto &ArgumentAsNode : ArgumentsAsNodes) {
    Children.push_back(llvm::cast<syntax::Expression>(ArgumentAsNode));
  }
  return Children;
}

std::vector<syntax::List::ElementAndDelimiter<syntax::Expression>>
syntax::CallArguments::getArgumentsAndCommas() {
  auto ArgumentsAsNodesAndCommas = getElementsAsNodesAndDelimiters();
  std::vector<syntax::List::ElementAndDelimiter<syntax::Expression>> Children;
  for (const auto &ArgumentAsNodeAndComma : ArgumentsAsNodesAndCommas) {
    Children.push_back(
        {llvm::cast<syntax::Expression>(ArgumentAsNodeAndComma.element),
         ArgumentAsNodeAndComma.delimiter});
  }
  return Children;
}

std::vector<syntax::SimpleDeclaration *>
syntax::ParameterDeclarationList::getParameterDeclarations() {
  auto ParametersAsNodes = getElementsAsNodes();
  std::vector<syntax::SimpleDeclaration *> Children;
  for (const auto &ParameterAsNode : ParametersAsNodes) {
    Children.push_back(llvm::cast<syntax::SimpleDeclaration>(ParameterAsNode));
  }
  return Children;
}

std::vector<syntax::List::ElementAndDelimiter<syntax::SimpleDeclaration>>
syntax::ParameterDeclarationList::getParametersAndCommas() {
  auto ParametersAsNodesAndCommas = getElementsAsNodesAndDelimiters();
  std::vector<syntax::List::ElementAndDelimiter<syntax::SimpleDeclaration>>
      Children;
  for (const auto &ParameterAsNodeAndComma : ParametersAsNodesAndCommas) {
    Children.push_back(
        {llvm::cast<syntax::SimpleDeclaration>(ParameterAsNodeAndComma.element),
         ParameterAsNodeAndComma.delimiter});
  }
  return Children;
}

syntax::Expression *syntax::MemberExpression::getObject() {
  return cast_or_null<syntax::Expression>(findChild(syntax::NodeRole::Object));
}

syntax::Leaf *syntax::MemberExpression::getTemplateKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::TemplateKeyword));
}

syntax::Leaf *syntax::MemberExpression::getAccessToken() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::AccessToken));
}

syntax::IdExpression *syntax::MemberExpression::getMember() {
  return cast_or_null<syntax::IdExpression>(
      findChild(syntax::NodeRole::Member));
}

syntax::NestedNameSpecifier *syntax::IdExpression::getQualifier() {
  return cast_or_null<syntax::NestedNameSpecifier>(
      findChild(syntax::NodeRole::Qualifier));
}

syntax::Leaf *syntax::IdExpression::getTemplateKeyword() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::TemplateKeyword));
}

syntax::UnqualifiedId *syntax::IdExpression::getUnqualifiedId() {
  return cast_or_null<syntax::UnqualifiedId>(
      findChild(syntax::NodeRole::UnqualifiedId));
}

syntax::Leaf *syntax::ParenExpression::getOpenParen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

syntax::Expression *syntax::ParenExpression::getSubExpression() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::SubExpression));
}

syntax::Leaf *syntax::ParenExpression::getCloseParen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Leaf *syntax::ThisExpression::getThisKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::LiteralExpression::getLiteralToken() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::LiteralToken));
}

syntax::Expression *syntax::BinaryOperatorExpression::getLhs() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::LeftHandSide));
}

syntax::Leaf *syntax::UnaryOperatorExpression::getOperatorToken() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OperatorToken));
}

syntax::Expression *syntax::UnaryOperatorExpression::getOperand() {
  return cast_or_null<syntax::Expression>(findChild(syntax::NodeRole::Operand));
}

syntax::Leaf *syntax::BinaryOperatorExpression::getOperatorToken() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OperatorToken));
}

syntax::Expression *syntax::BinaryOperatorExpression::getRhs() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::RightHandSide));
}

syntax::Expression *syntax::CallExpression::getCallee() {
  return cast_or_null<syntax::Expression>(findChild(syntax::NodeRole::Callee));
}

syntax::Leaf *syntax::CallExpression::getOpenParen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

syntax::CallArguments *syntax::CallExpression::getArguments() {
  return cast_or_null<syntax::CallArguments>(
      findChild(syntax::NodeRole::Arguments));
}

syntax::Leaf *syntax::CallExpression::getCloseParen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Leaf *syntax::SwitchStatement::getSwitchKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::SwitchStatement::getBody() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::CaseStatement::getCaseKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Expression *syntax::CaseStatement::getCaseValue() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::CaseValue));
}

syntax::Statement *syntax::CaseStatement::getBody() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::DefaultStatement::getDefaultKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::DefaultStatement::getBody() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::IfStatement::getIfKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::IfStatement::getThenStatement() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::ThenStatement));
}

syntax::Leaf *syntax::IfStatement::getElseKeyword() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::ElseKeyword));
}

syntax::Statement *syntax::IfStatement::getElseStatement() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::ElseStatement));
}

syntax::Leaf *syntax::ForStatement::getForKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::ForStatement::getBody() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::WhileStatement::getWhileKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::WhileStatement::getBody() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Leaf *syntax::ContinueStatement::getContinueKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::BreakStatement::getBreakKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::ReturnStatement::getReturnKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Expression *syntax::ReturnStatement::getReturnValue() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::ReturnValue));
}

syntax::Leaf *syntax::RangeBasedForStatement::getForKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Statement *syntax::RangeBasedForStatement::getBody() {
  return cast_or_null<syntax::Statement>(
      findChild(syntax::NodeRole::BodyStatement));
}

syntax::Expression *syntax::ExpressionStatement::getExpression() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::Expression));
}

syntax::Leaf *syntax::CompoundStatement::getLbrace() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

std::vector<syntax::Statement *> syntax::CompoundStatement::getStatements() {
  std::vector<syntax::Statement *> Children;
  for (auto *C = getFirstChild(); C; C = C->getNextSibling()) {
    assert(C->getRole() == syntax::NodeRole::Statement);
    Children.push_back(cast<syntax::Statement>(C));
  }
  return Children;
}

syntax::Leaf *syntax::CompoundStatement::getRbrace() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Expression *syntax::StaticAssertDeclaration::getCondition() {
  return cast_or_null<syntax::Expression>(
      findChild(syntax::NodeRole::Condition));
}

syntax::Expression *syntax::StaticAssertDeclaration::getMessage() {
  return cast_or_null<syntax::Expression>(findChild(syntax::NodeRole::Message));
}

std::vector<syntax::SimpleDeclarator *>
syntax::SimpleDeclaration::getDeclarators() {
  std::vector<syntax::SimpleDeclarator *> Children;
  for (auto *C = getFirstChild(); C; C = C->getNextSibling()) {
    if (C->getRole() == syntax::NodeRole::Declarator)
      Children.push_back(cast<syntax::SimpleDeclarator>(C));
  }
  return Children;
}

syntax::Leaf *syntax::TemplateDeclaration::getTemplateKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Declaration *syntax::TemplateDeclaration::getDeclaration() {
  return cast_or_null<syntax::Declaration>(
      findChild(syntax::NodeRole::Declaration));
}

syntax::Leaf *syntax::ExplicitTemplateInstantiation::getTemplateKeyword() {
  return cast_or_null<syntax::Leaf>(
      findChild(syntax::NodeRole::IntroducerKeyword));
}

syntax::Leaf *syntax::ExplicitTemplateInstantiation::getExternKeyword() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::ExternKeyword));
}

syntax::Declaration *syntax::ExplicitTemplateInstantiation::getDeclaration() {
  return cast_or_null<syntax::Declaration>(
      findChild(syntax::NodeRole::Declaration));
}

syntax::Leaf *syntax::ParenDeclarator::getLparen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

syntax::Leaf *syntax::ParenDeclarator::getRparen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Leaf *syntax::ArraySubscript::getLbracket() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

syntax::Expression *syntax::ArraySubscript::getSize() {
  return cast_or_null<syntax::Expression>(findChild(syntax::NodeRole::Size));
}

syntax::Leaf *syntax::ArraySubscript::getRbracket() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::Leaf *syntax::TrailingReturnType::getArrowToken() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::ArrowToken));
}

syntax::SimpleDeclarator *syntax::TrailingReturnType::getDeclarator() {
  return cast_or_null<syntax::SimpleDeclarator>(
      findChild(syntax::NodeRole::Declarator));
}

syntax::Leaf *syntax::ParametersAndQualifiers::getLparen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::OpenParen));
}

syntax::ParameterDeclarationList *
syntax::ParametersAndQualifiers::getParameters() {
  return cast_or_null<syntax::ParameterDeclarationList>(
      findChild(syntax::NodeRole::Parameters));
}

syntax::Leaf *syntax::ParametersAndQualifiers::getRparen() {
  return cast_or_null<syntax::Leaf>(findChild(syntax::NodeRole::CloseParen));
}

syntax::TrailingReturnType *
syntax::ParametersAndQualifiers::getTrailingReturn() {
  return cast_or_null<syntax::TrailingReturnType>(
      findChild(syntax::NodeRole::TrailingReturn));
}
