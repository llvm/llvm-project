/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Syntax tree node list                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Nodes.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


// Forward-declare node types so we don't have to carefully sequence definitions.
class Node;
class Leaf;
class Tree;
class ArraySubscript;
class Declaration;
class EmptyDeclaration;
class ExplicitTemplateInstantiation;
class LinkageSpecificationDeclaration;
class NamespaceAliasDefinition;
class NamespaceDefinition;
class SimpleDeclaration;
class StaticAssertDeclaration;
class TemplateDeclaration;
class TypeAliasDeclaration;
class UnknownDeclaration;
class UsingDeclaration;
class UsingNamespaceDirective;
class Declarator;
class ParenDeclarator;
class SimpleDeclarator;
class Expression;
class BinaryOperatorExpression;
class CallExpression;
class IdExpression;
class LiteralExpression;
class BoolLiteralExpression;
class CharacterLiteralExpression;
class CxxNullPtrExpression;
class FloatingLiteralExpression;
class IntegerLiteralExpression;
class StringLiteralExpression;
class UserDefinedLiteralExpression;
class CharUserDefinedLiteralExpression;
class FloatUserDefinedLiteralExpression;
class IntegerUserDefinedLiteralExpression;
class StringUserDefinedLiteralExpression;
class MemberExpression;
class ParenExpression;
class ThisExpression;
class UnknownExpression;
class List;
class CallArguments;
class DeclaratorList;
class NestedNameSpecifier;
class ParameterDeclarationList;
class MemberPointer;
class NameSpecifier;
class DecltypeNameSpecifier;
class GlobalNameSpecifier;
class IdentifierNameSpecifier;
class SimpleTemplateNameSpecifier;
class ParametersAndQualifiers;
class Statement;
class BreakStatement;
class CaseStatement;
class CompoundStatement;
class ContinueStatement;
class DeclarationStatement;
class DefaultStatement;
class EmptyStatement;
class ExpressionStatement;
class ForStatement;
class IfStatement;
class RangeBasedForStatement;
class ReturnStatement;
class SwitchStatement;
class UnknownStatement;
class WhileStatement;
class TrailingReturnType;
class TranslationUnit;
class UnaryOperatorExpression;
class PostfixUnaryOperatorExpression;
class PrefixUnaryOperatorExpression;
class UnqualifiedId;

// Node definitions

/// A base class for all expressions. Note that expressions are not statements,
/// even though they are in clang.
class Expression : public Tree {
protected:
  Expression(NodeKind K) : Tree(K) {}
public:
  static bool classof(const Node *N);
};

/// A function call. C++ [expr.call]
/// call-expression:
///   expression '(' call-arguments ')'
/// e.g `f(1, '2')` or `this->Base::f()`
class CallExpression final : public Expression {
public:
  CallExpression() : Expression(NodeKind::CallExpression) {}
  Expression *getCallee()  {
    return llvm::cast_or_null<Expression>(findChild(NodeRole::Callee));
  }
  const Expression *getCallee() const  {
    return llvm::cast_or_null<Expression>(findChild(NodeRole::Callee));
  }
  Leaf *getOpenParen()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::OpenParen));
  }
  const Leaf *getOpenParen() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::OpenParen));
  }
  CallArguments *getArguments()  {
    return llvm::cast_or_null<CallArguments>(findChild(NodeRole::Arguments));
  }
  const CallArguments *getArguments() const  {
    return llvm::cast_or_null<CallArguments>(findChild(NodeRole::Arguments));
  }
  Leaf *getCloseParen()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::CloseParen));
  }
  const Leaf *getCloseParen() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::CloseParen));
  }
  static bool classof(const Node *N);
};

/// Models an `id-expression`, e.g. `std::vector<int>::size`.
/// C++ [expr.prim.id]
/// id-expression:
///   unqualified-id
///   qualified-id
/// qualified-id:
///   nested-name-specifier template_opt unqualified-id
class IdExpression final : public Expression {
public:
  IdExpression() : Expression(NodeKind::IdExpression) {}
  NestedNameSpecifier *getQualifier()  {
    return llvm::cast_or_null<NestedNameSpecifier>(findChild(NodeRole::Qualifier));
  }
  const NestedNameSpecifier *getQualifier() const  {
    return llvm::cast_or_null<NestedNameSpecifier>(findChild(NodeRole::Qualifier));
  }
  Leaf *getTemplateKeyword()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::TemplateKeyword));
  }
  const Leaf *getTemplateKeyword() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::TemplateKeyword));
  }
  UnqualifiedId *getUnqualifiedId()  {
    return llvm::cast_or_null<UnqualifiedId>(findChild(NodeRole::UnqualifiedId));
  }
  const UnqualifiedId *getUnqualifiedId() const  {
    return llvm::cast_or_null<UnqualifiedId>(findChild(NodeRole::UnqualifiedId));
  }
  static bool classof(const Node *N);
};

/// Expression for literals. C++ [lex.literal]
class LiteralExpression : public Expression {
protected:
  LiteralExpression(NodeKind K) : Expression(K) {}
public:
  static bool classof(const Node *N);
};

/// Expression for boolean literals. C++ [lex.bool]
class BoolLiteralExpression final : public LiteralExpression {
public:
  BoolLiteralExpression() : LiteralExpression(NodeKind::BoolLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for character literals. C++ [lex.ccon]
class CharacterLiteralExpression final : public LiteralExpression {
public:
  CharacterLiteralExpression() : LiteralExpression(NodeKind::CharacterLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for the `nullptr` literal. C++ [lex.nullptr]
class CxxNullPtrExpression final : public LiteralExpression {
public:
  CxxNullPtrExpression() : LiteralExpression(NodeKind::CxxNullPtrExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for floating-point literals. C++ [lex.fcon]
class FloatingLiteralExpression final : public LiteralExpression {
public:
  FloatingLiteralExpression() : LiteralExpression(NodeKind::FloatingLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for integer literals. C++ [lex.icon]
class IntegerLiteralExpression final : public LiteralExpression {
public:
  IntegerLiteralExpression() : LiteralExpression(NodeKind::IntegerLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for string-literals. C++ [lex.string]
class StringLiteralExpression final : public LiteralExpression {
public:
  StringLiteralExpression() : LiteralExpression(NodeKind::StringLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for user-defined literal. C++ [lex.ext]
/// user-defined-literal:
///   user-defined-integer-literal
///   user-defined-floating-point-literal
///   user-defined-string-literal
///   user-defined-character-literal
class UserDefinedLiteralExpression : public LiteralExpression {
protected:
  UserDefinedLiteralExpression(NodeKind K) : LiteralExpression(K) {}
public:
  static bool classof(const Node *N);
};

/// Expression for user-defined-character-literal. C++ [lex.ext]
class CharUserDefinedLiteralExpression final : public UserDefinedLiteralExpression {
public:
  CharUserDefinedLiteralExpression() : UserDefinedLiteralExpression(NodeKind::CharUserDefinedLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for user-defined-floating-point-literal. C++ [lex.ext]
class FloatUserDefinedLiteralExpression final : public UserDefinedLiteralExpression {
public:
  FloatUserDefinedLiteralExpression() : UserDefinedLiteralExpression(NodeKind::FloatUserDefinedLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for user-defined-integer-literal. C++ [lex.ext]
class IntegerUserDefinedLiteralExpression final : public UserDefinedLiteralExpression {
public:
  IntegerUserDefinedLiteralExpression() : UserDefinedLiteralExpression(NodeKind::IntegerUserDefinedLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Expression for user-defined-string-literal. C++ [lex.ext]
class StringUserDefinedLiteralExpression final : public UserDefinedLiteralExpression {
public:
  StringUserDefinedLiteralExpression() : UserDefinedLiteralExpression(NodeKind::StringUserDefinedLiteralExpression) {}
  Leaf *getLiteralToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  const Leaf *getLiteralToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::LiteralToken));
  }
  static bool classof(const Node *N);
};

/// Models a class member access. C++ [expr.ref]
/// member-expression:
///   expression -> template_opt id-expression
///   expression .  template_opt id-expression
/// e.g. `x.a`, `xp->a`
/// 
/// Note: An implicit member access inside a class, i.e. `a` instead of
/// `this->a`, is an `id-expression`.
class MemberExpression final : public Expression {
public:
  MemberExpression() : Expression(NodeKind::MemberExpression) {}
  Expression *getObject()  {
    return llvm::cast_or_null<Expression>(findChild(NodeRole::Object));
  }
  const Expression *getObject() const  {
    return llvm::cast_or_null<Expression>(findChild(NodeRole::Object));
  }
  Leaf *getAccessToken()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::AccessToken));
  }
  const Leaf *getAccessToken() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::AccessToken));
  }
  Leaf *getTemplateKeyword()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::TemplateKeyword));
  }
  const Leaf *getTemplateKeyword() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::TemplateKeyword));
  }
  IdExpression *getMember()  {
    return llvm::cast_or_null<IdExpression>(findChild(NodeRole::Member));
  }
  const IdExpression *getMember() const  {
    return llvm::cast_or_null<IdExpression>(findChild(NodeRole::Member));
  }
  static bool classof(const Node *N);
};

/// Models a parenthesized expression `(E)`. C++ [expr.prim.paren]
/// e.g. `(3 + 2)` in `a = 1 + (3 + 2);`
class ParenExpression final : public Expression {
public:
  ParenExpression() : Expression(NodeKind::ParenExpression) {}
  Leaf *getOpenParen()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::OpenParen));
  }
  const Leaf *getOpenParen() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::OpenParen));
  }
  Expression *getSubExpression()  {
    return llvm::cast_or_null<Expression>(findChild(NodeRole::SubExpression));
  }
  const Expression *getSubExpression() const  {
    return llvm::cast_or_null<Expression>(findChild(NodeRole::SubExpression));
  }
  Leaf *getCloseParen()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::CloseParen));
  }
  const Leaf *getCloseParen() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::CloseParen));
  }
  static bool classof(const Node *N);
};

/// Models a this expression `this`. C++ [expr.prim.this]
class ThisExpression final : public Expression {
public:
  ThisExpression() : Expression(NodeKind::ThisExpression) {}
  Leaf *getIntroducerKeyword()  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::IntroducerKeyword));
  }
  const Leaf *getIntroducerKeyword() const  {
    return llvm::cast_or_null<Leaf>(findChild(NodeRole::IntroducerKeyword));
  }
  static bool classof(const Node *N);
};

/// A sequence of these specifiers make a `nested-name-specifier`.
/// e.g. the `std` or `vector<int>` in `std::vector<int>::size`.
class NameSpecifier : public Tree {
protected:
  NameSpecifier(NodeKind K) : Tree(K) {}
public:
  static bool classof(const Node *N);
};

/// A name specifier holding a decltype, of the form: `decltype ( expression ) `
/// e.g. the `decltype(s)` in `decltype(s)::size`.
class DecltypeNameSpecifier final : public NameSpecifier {
public:
  DecltypeNameSpecifier() : NameSpecifier(NodeKind::DecltypeNameSpecifier) {}
  static bool classof(const Node *N);
};

/// The global namespace name specifier, this specifier doesn't correspond to a
/// token instead an absence of tokens before a `::` characterizes it, in
/// `::std::vector<int>` it would be characterized by the absence of a token
/// before the first `::`
class GlobalNameSpecifier final : public NameSpecifier {
public:
  GlobalNameSpecifier() : NameSpecifier(NodeKind::GlobalNameSpecifier) {}
  static bool classof(const Node *N);
};

/// A identifier name specifier, of the form `identifier`
/// e.g. the `std` in `std::vector<int>::size`.
class IdentifierNameSpecifier final : public NameSpecifier {
public:
  IdentifierNameSpecifier() : NameSpecifier(NodeKind::IdentifierNameSpecifier) {}
  static bool classof(const Node *N);
};

/// A name specifier with a simple-template-id, of the form `template_opt
/// identifier < template-args >` e.g. the `vector<int>` in
/// `std::vector<int>::size`.
class SimpleTemplateNameSpecifier final : public NameSpecifier {
public:
  SimpleTemplateNameSpecifier() : NameSpecifier(NodeKind::SimpleTemplateNameSpecifier) {}
  static bool classof(const Node *N);
};

/// A root node for a translation unit. Parent is always null.
class TranslationUnit final : public Tree {
public:
  TranslationUnit() : Tree(NodeKind::TranslationUnit) {}
  static bool classof(const Node *N);
};

