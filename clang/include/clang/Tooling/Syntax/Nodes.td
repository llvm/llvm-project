//===- Nodes.td - Node types in the Syntax Tree grammar -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines concrete nodes in the syntax tree.
// The archetypes they fall into (Sequence, List etc) are defined in Syntax.td.
//
// The C++ classes for the archetypes themselves are written by hand, and the
// concrete node classes will be generated. Migration to TableGen is not
// complete, so currently there is a mix of generated and hand-authored code.
//
//===----------------------------------------------------------------------===//

include "clang/Tooling/Syntax/Syntax.td"

def Node : External<?> {}
def Leaf : External<Node> {}
def Tree : External<Node> {}

def TranslationUnit : External<Tree> {}

def Expression : External<Tree> {}
def UnknownExpression : External<Expression> {}
def UnaryOperatorExpression : External<Tree> {}
def PrefixUnaryOperatorExpression : External<UnaryOperatorExpression> {}
def PostfixUnaryOperatorExpression : External<UnaryOperatorExpression> {}
def BinaryOperatorExpression : External<Expression> {}
def ParenExpression : External<Expression> {}
def LiteralExpression : External<Expression> {}
def IntegerLiteralExpression : External<LiteralExpression> {}
def CharacterLiteralExpression : External<LiteralExpression> {}
def FloatingLiteralExpression : External<LiteralExpression> {}
def StringLiteralExpression : External<LiteralExpression> {}
def BoolLiteralExpression : External<LiteralExpression> {}
def CxxNullPtrExpression : External<LiteralExpression> {}
def UserDefinedLiteralExpression : External<LiteralExpression> {}
def IntegerUserDefinedLiteralExpression : External<UserDefinedLiteralExpression> {}
def FloatUserDefinedLiteralExpression : External<UserDefinedLiteralExpression> {}
def CharUserDefinedLiteralExpression : External<UserDefinedLiteralExpression> {}
def StringUserDefinedLiteralExpression : External<UserDefinedLiteralExpression> {}
def IdExpression : External<Expression> {}
def MemberExpression : External<Expression> {}
def ThisExpression : External<Expression> {}
def CallExpression : External<Expression> {}

// Statements.
def Statement : External<Tree> {}
def UnknownStatement : External<Statement> {}
def DeclarationStatement : External<Statement> {}
def EmptyStatement : External<Statement> {}
def SwitchStatement : External<Statement> {}
def CaseStatement : External<Statement> {}
def DefaultStatement : External<Statement> {}
def IfStatement : External<Statement> {}
def ForStatement : External<Statement> {}
def WhileStatement : External<Statement> {}
def ContinueStatement : External<Statement> {}
def BreakStatement : External<Statement> {}
def ReturnStatement : External<Statement> {}
def RangeBasedForStatement : External<Statement> {}
def ExpressionStatement : External<Statement> {}
def CompoundStatement : External<Statement> {}

// Declarations.
def Declaration : External<Tree> {}
def UnknownDeclaration : External<Declaration> {}
def EmptyDeclaration : External<Declaration> {}
def StaticAssertDeclaration : External<Declaration> {}
def LinkageSpecificationDeclaration : External<Declaration> {}
def SimpleDeclaration : External<Declaration> {}
def TemplateDeclaration : External<Declaration> {}
def ExplicitTemplateInstantiation : External<Declaration> {}
def NamespaceDefinition : External<Declaration> {}
def NamespaceAliasDefinition : External<Declaration> {}
def UsingNamespaceDirective : External<Declaration> {}
def UsingDeclaration : External<Declaration> {}
def TypeAliasDeclaration : External<Declaration> {}

// Declarators.
def Declarator : External<Tree> {}
def SimpleDeclarator : External<Declarator> {}
def ParenDeclarator : External<Declarator> {}

def ArraySubscript : External<Tree> {}
def TrailingReturnType : External<Tree> {}
def ParametersAndQualifiers : External<Tree> {}
def MemberPointer : External<Tree> {}
def UnqualifiedId : External<Tree> {}

// Lists
def List : External<Tree> {}
def DeclaratorList : External<List> {}
def ParameterDeclarationList : External<List> {}
def CallArguments : External<List> {}
def NestedNameSpecifier : External<List> {}

// Name Specifiers.
def NameSpecifier : External<Tree> {}
def GlobalNameSpecifier : External<NameSpecifier> {}
def DecltypeNameSpecifier : External<NameSpecifier> {}
def IdentifierNameSpecifier : External<NameSpecifier> {}
def SimpleTemplateNameSpecifier : External<NameSpecifier> {}
