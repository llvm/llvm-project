//===-- lib/Parser/type-parsers.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_TYPE_PARSERS_H_
#define FORTRAN_PARSER_TYPE_PARSERS_H_

#include "flang/Parser/instrumented-parser.h"
#include "flang/Parser/parse-tree.h"
#include <optional>

namespace Fortran::parser {

// Many parsers in the grammar are defined as instances of this Parser<>
// class template, i.e. as the anonymous sole parser for a given type.
// This usage requires that their Parse() member functions be defined
// separately, typically with a parsing expression wrapped up in an
// TYPE_PARSER() macro call.
template <typename A> struct Parser {
  using resultType = A;
  constexpr Parser() {}
  constexpr Parser(const Parser &) = default;
  static std::optional<resultType> Parse(ParseState &);
};

#define CONTEXT_PARSER(contextText, pexpr) \
  instrumented((contextText), inContext((contextText), (pexpr)))

// To allow use of the Fortran grammar (or parts of it) outside the
// context of constructing the actual parser.
#define TYPE_PARSER(pexpr)
#define TYPE_CONTEXT_PARSER(context, pexpr)

// Some specializations of Parser<> are used multiple times, or are
// of some special importance, so we instantiate them once here and
// give them names rather than referencing them as anonymous Parser<T>{}
// objects in the right-hand sides of productions.
constexpr Parser<Program> program; //  R501 - the "top level" production
constexpr Parser<SpecificationPart> specificationPart; //  R504
constexpr Parser<ImplicitPart> implicitPart; //  R505
constexpr Parser<DeclarationConstruct> declarationConstruct; //  R507
constexpr Parser<SpecificationConstruct> specificationConstruct; //  R508
constexpr Parser<ExecutionPart> executionPart; //  R509
constexpr Parser<ExecutionPartConstruct> executionPartConstruct; //  R510
constexpr Parser<InternalSubprogramPart> internalSubprogramPart; //  R511
constexpr Parser<ActionStmt> actionStmt; // R515
constexpr Parser<Name> name; // R603
constexpr Parser<LiteralConstant> literalConstant; // R605
constexpr Parser<NamedConstant> namedConstant; // R606
constexpr Parser<TypeParamValue> typeParamValue; // R701
constexpr Parser<TypeSpec> typeSpec; // R702
constexpr Parser<DeclarationTypeSpec> declarationTypeSpec; // R703
constexpr Parser<IntrinsicTypeSpec> intrinsicTypeSpec; // R704
constexpr Parser<IntegerTypeSpec> integerTypeSpec; // R705
constexpr Parser<KindSelector> kindSelector; // R706
constexpr Parser<SignedIntLiteralConstant> signedIntLiteralConstant; // R707
constexpr Parser<IntLiteralConstant> intLiteralConstant; // R708
constexpr Parser<KindParam> kindParam; // R709
constexpr Parser<RealLiteralConstant> realLiteralConstant; // R714
constexpr Parser<CharLength> charLength; // R723
constexpr Parser<CharLiteralConstant> charLiteralConstant; // R724
constexpr Parser<CharLiteralConstantSubstring> charLiteralConstantSubstring;
constexpr Parser<Initialization> initialization; // R743 & R805
constexpr Parser<DerivedTypeSpec> derivedTypeSpec; // R754
constexpr Parser<TypeDeclarationStmt> typeDeclarationStmt; // R801
constexpr Parser<NullInit> nullInit; // R806
constexpr Parser<AccessSpec> accessSpec; // R807
constexpr Parser<LanguageBindingSpec> languageBindingSpec; // R808, R1528
constexpr Parser<EntityDecl> entityDecl; // R803
constexpr Parser<CoarraySpec> coarraySpec; // R809
constexpr Parser<ArraySpec> arraySpec; // R815
constexpr Parser<ComponentArraySpec> componentArraySpec;
constexpr Parser<ExplicitShapeSpec> explicitShapeSpec; // R816
constexpr Parser<DeferredShapeSpecList> deferredShapeSpecList; // R820
constexpr Parser<AssumedImpliedSpec> assumedImpliedSpec; // R821
constexpr Parser<IntentSpec> intentSpec; // R826
constexpr Parser<DataStmt> dataStmt; // R837
constexpr Parser<DataImpliedDo> dataImpliedDo; // R840
constexpr Parser<ParameterStmt> parameterStmt; // R851
constexpr Parser<OldParameterStmt> oldParameterStmt;
constexpr Parser<Designator> designator; // R901
constexpr Parser<Variable> variable; // R902
constexpr Parser<Substring> substring; // R908
constexpr Parser<DataRef> dataRef; // R911, R914, R917
constexpr Parser<StructureComponent> structureComponent; // R913
constexpr Parser<SubscriptTriplet> subscriptTriplet; // R921
constexpr Parser<AllocateStmt> allocateStmt; // R927
constexpr Parser<StatVariable> statVariable; // R929
constexpr Parser<StatOrErrmsg> statOrErrmsg; // R942 & R1165
constexpr Parser<DefinedOpName> definedOpName; // R1003, R1023, R1414, & R1415
constexpr Parser<Expr> expr; // R1022
constexpr Parser<SpecificationExpr> specificationExpr; // R1028
constexpr Parser<AssignmentStmt> assignmentStmt; // R1032
constexpr Parser<PointerAssignmentStmt> pointerAssignmentStmt; // R1033
constexpr Parser<WhereStmt> whereStmt; // R1041, R1045, R1046
constexpr Parser<WhereConstruct> whereConstruct; // R1042
constexpr Parser<WhereBodyConstruct> whereBodyConstruct; // R1044
constexpr Parser<ForallConstruct> forallConstruct; // R1050
constexpr Parser<ForallAssignmentStmt> forallAssignmentStmt; // R1053
constexpr Parser<ForallStmt> forallStmt; // R1055
constexpr Parser<Selector> selector; // R1105
constexpr Parser<EndSelectStmt> endSelectStmt; // R1143 & R1151 & R1155
constexpr Parser<ConcurrentHeader> concurrentHeader; // R1125
constexpr Parser<IoUnit> ioUnit; // R1201, R1203
constexpr Parser<FileUnitNumber> fileUnitNumber; // R1202
constexpr Parser<IoControlSpec> ioControlSpec; // R1213, R1214
constexpr Parser<Format> format; // R1215
constexpr Parser<InputItem> inputItem; // R1216
constexpr Parser<OutputItem> outputItem; // R1217
constexpr Parser<InputImpliedDo> inputImpliedDo; // R1218, R1219
constexpr Parser<OutputImpliedDo> outputImpliedDo; // R1218, R1219
constexpr Parser<PositionOrFlushSpec> positionOrFlushSpec; // R1227 & R1229
constexpr Parser<FormatStmt> formatStmt; // R1301
constexpr Parser<InterfaceBlock> interfaceBlock; // R1501
constexpr Parser<GenericSpec> genericSpec; // R1508
constexpr Parser<ProcInterface> procInterface; // R1513
constexpr Parser<ProcDecl> procDecl; // R1515
constexpr Parser<FunctionReference> functionReference; // R1520
constexpr Parser<ActualArgSpec> actualArgSpec; // R1523
constexpr Parser<PrefixSpec> prefixSpec; // R1527
constexpr Parser<FunctionSubprogram> functionSubprogram; // R1529
constexpr Parser<FunctionStmt> functionStmt; // R1530
constexpr Parser<Suffix> suffix; // R1532
constexpr Parser<EndFunctionStmt> endFunctionStmt; // R1533
constexpr Parser<SubroutineSubprogram> subroutineSubprogram; // R1534
constexpr Parser<SubroutineStmt> subroutineStmt; // R1535
constexpr Parser<DummyArg> dummyArg; // R1536
constexpr Parser<EndSubroutineStmt> endSubroutineStmt; // R1537
constexpr Parser<EntryStmt> entryStmt; // R1541
constexpr Parser<ContainsStmt> containsStmt; // R1543
constexpr Parser<CompilerDirective> compilerDirective;
constexpr Parser<OpenACCConstruct> openaccConstruct;
constexpr Parser<OpenACCDeclarativeConstruct> openaccDeclarativeConstruct;
constexpr Parser<OpenMPConstruct> openmpConstruct;
constexpr Parser<OpenMPDeclarativeConstruct> openmpDeclarativeConstruct;
constexpr Parser<OmpEndLoopDirective> ompEndLoopDirective;
constexpr Parser<IntrinsicVectorTypeSpec> intrinsicVectorTypeSpec; // Extension
constexpr Parser<VectorTypeSpec> vectorTypeSpec; // Extension
constexpr Parser<UnsignedTypeSpec> unsignedTypeSpec; // Extension
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_TYPE_PARSERS_H_
