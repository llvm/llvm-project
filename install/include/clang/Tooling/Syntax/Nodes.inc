/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Syntax tree node list                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Nodes.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifndef NODE
#define NODE(Kind, Base)
#endif

#ifndef CONCRETE_NODE
#define CONCRETE_NODE(Kind, Base) NODE(Kind, Base)
#endif

#ifndef ABSTRACT_NODE
#define ABSTRACT_NODE(Kind, Base, First, Last) NODE(Kind, Base)
#endif

CONCRETE_NODE(Leaf,Node)
ABSTRACT_NODE(Tree,Node,ArraySubscript,UnqualifiedId)
CONCRETE_NODE(ArraySubscript,Tree)
ABSTRACT_NODE(Declaration,Tree,EmptyDeclaration,UsingNamespaceDirective)
CONCRETE_NODE(EmptyDeclaration,Declaration)
CONCRETE_NODE(ExplicitTemplateInstantiation,Declaration)
CONCRETE_NODE(LinkageSpecificationDeclaration,Declaration)
CONCRETE_NODE(NamespaceAliasDefinition,Declaration)
CONCRETE_NODE(NamespaceDefinition,Declaration)
CONCRETE_NODE(SimpleDeclaration,Declaration)
CONCRETE_NODE(StaticAssertDeclaration,Declaration)
CONCRETE_NODE(TemplateDeclaration,Declaration)
CONCRETE_NODE(TypeAliasDeclaration,Declaration)
CONCRETE_NODE(UnknownDeclaration,Declaration)
CONCRETE_NODE(UsingDeclaration,Declaration)
CONCRETE_NODE(UsingNamespaceDirective,Declaration)
ABSTRACT_NODE(Declarator,Tree,ParenDeclarator,SimpleDeclarator)
CONCRETE_NODE(ParenDeclarator,Declarator)
CONCRETE_NODE(SimpleDeclarator,Declarator)
ABSTRACT_NODE(Expression,Tree,BinaryOperatorExpression,UnknownExpression)
CONCRETE_NODE(BinaryOperatorExpression,Expression)
CONCRETE_NODE(CallExpression,Expression)
CONCRETE_NODE(IdExpression,Expression)
ABSTRACT_NODE(LiteralExpression,Expression,BoolLiteralExpression,StringUserDefinedLiteralExpression)
CONCRETE_NODE(BoolLiteralExpression,LiteralExpression)
CONCRETE_NODE(CharacterLiteralExpression,LiteralExpression)
CONCRETE_NODE(CxxNullPtrExpression,LiteralExpression)
CONCRETE_NODE(FloatingLiteralExpression,LiteralExpression)
CONCRETE_NODE(IntegerLiteralExpression,LiteralExpression)
CONCRETE_NODE(StringLiteralExpression,LiteralExpression)
ABSTRACT_NODE(UserDefinedLiteralExpression,LiteralExpression,CharUserDefinedLiteralExpression,StringUserDefinedLiteralExpression)
CONCRETE_NODE(CharUserDefinedLiteralExpression,UserDefinedLiteralExpression)
CONCRETE_NODE(FloatUserDefinedLiteralExpression,UserDefinedLiteralExpression)
CONCRETE_NODE(IntegerUserDefinedLiteralExpression,UserDefinedLiteralExpression)
CONCRETE_NODE(StringUserDefinedLiteralExpression,UserDefinedLiteralExpression)
CONCRETE_NODE(MemberExpression,Expression)
CONCRETE_NODE(ParenExpression,Expression)
CONCRETE_NODE(ThisExpression,Expression)
CONCRETE_NODE(UnknownExpression,Expression)
ABSTRACT_NODE(List,Tree,CallArguments,ParameterDeclarationList)
CONCRETE_NODE(CallArguments,List)
CONCRETE_NODE(DeclaratorList,List)
CONCRETE_NODE(NestedNameSpecifier,List)
CONCRETE_NODE(ParameterDeclarationList,List)
CONCRETE_NODE(MemberPointer,Tree)
ABSTRACT_NODE(NameSpecifier,Tree,DecltypeNameSpecifier,SimpleTemplateNameSpecifier)
CONCRETE_NODE(DecltypeNameSpecifier,NameSpecifier)
CONCRETE_NODE(GlobalNameSpecifier,NameSpecifier)
CONCRETE_NODE(IdentifierNameSpecifier,NameSpecifier)
CONCRETE_NODE(SimpleTemplateNameSpecifier,NameSpecifier)
CONCRETE_NODE(ParametersAndQualifiers,Tree)
ABSTRACT_NODE(Statement,Tree,BreakStatement,WhileStatement)
CONCRETE_NODE(BreakStatement,Statement)
CONCRETE_NODE(CaseStatement,Statement)
CONCRETE_NODE(CompoundStatement,Statement)
CONCRETE_NODE(ContinueStatement,Statement)
CONCRETE_NODE(DeclarationStatement,Statement)
CONCRETE_NODE(DefaultStatement,Statement)
CONCRETE_NODE(EmptyStatement,Statement)
CONCRETE_NODE(ExpressionStatement,Statement)
CONCRETE_NODE(ForStatement,Statement)
CONCRETE_NODE(IfStatement,Statement)
CONCRETE_NODE(RangeBasedForStatement,Statement)
CONCRETE_NODE(ReturnStatement,Statement)
CONCRETE_NODE(SwitchStatement,Statement)
CONCRETE_NODE(UnknownStatement,Statement)
CONCRETE_NODE(WhileStatement,Statement)
CONCRETE_NODE(TrailingReturnType,Tree)
CONCRETE_NODE(TranslationUnit,Tree)
ABSTRACT_NODE(UnaryOperatorExpression,Tree,PostfixUnaryOperatorExpression,PrefixUnaryOperatorExpression)
CONCRETE_NODE(PostfixUnaryOperatorExpression,UnaryOperatorExpression)
CONCRETE_NODE(PrefixUnaryOperatorExpression,UnaryOperatorExpression)
CONCRETE_NODE(UnqualifiedId,Tree)

#undef NODE
#undef CONCRETE_NODE
#undef ABSTRACT_NODE
