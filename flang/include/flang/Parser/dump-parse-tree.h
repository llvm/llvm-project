//===-- include/flang/Parser/dump-parse-tree.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_DUMP_PARSE_TREE_H_
#define FORTRAN_PARSER_DUMP_PARSE_TREE_H_

#include "format-specification.h"
#include "parse-tree-visitor.h"
#include "parse-tree.h"
#include "tools.h"
#include "unparse.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <type_traits>

namespace Fortran::parser {

//
// Dump the Parse Tree hierarchy of any node 'x' of the parse tree.
//

class ParseTreeDumper {
public:
  explicit ParseTreeDumper(llvm::raw_ostream &out,
      const AnalyzedObjectsAsFortran *asFortran = nullptr)
      : out_(out), asFortran_{asFortran} {}

  static constexpr const char *GetNodeName(const char *) { return "char *"; }
#define NODE_NAME(T, N) \
  static constexpr const char *GetNodeName(const T &) { return N; }
#define NODE_ENUM(T, E) \
  static std::string GetNodeName(const T::E &x) { \
    return #E " = "s + T::EnumToString(x); \
  }
#define NODE(T1, T2) NODE_NAME(T1::T2, #T2)
  NODE_NAME(bool, "bool")
  NODE_NAME(int, "int")
  NODE(std, string)
  NODE(std, int64_t)
  NODE(std, uint64_t)
  NODE(format, ControlEditDesc)
  NODE(format::ControlEditDesc, Kind)
  NODE(format, DerivedTypeDataEditDesc)
  NODE(format, FormatItem)
  NODE(format, FormatSpecification)
  NODE(format, IntrinsicTypeDataEditDesc)
  NODE(format::IntrinsicTypeDataEditDesc, Kind)
  NODE(parser, Abstract)
  NODE(parser, AccAtomicCapture)
  NODE(AccAtomicCapture, Stmt1)
  NODE(AccAtomicCapture, Stmt2)
  NODE(parser, AccAtomicRead)
  NODE(parser, AccAtomicUpdate)
  NODE(parser, AccAtomicWrite)
  NODE(parser, AccBeginBlockDirective)
  NODE(parser, AccBeginCombinedDirective)
  NODE(parser, AccBeginLoopDirective)
  NODE(parser, AccBlockDirective)
  NODE(parser, AccClause)
  NODE(AccClause, Auto)
  NODE(AccClause, Async)
  NODE(AccClause, Attach)
  NODE(AccClause, Bind)
  NODE(AccClause, Capture)
  NODE(AccClause, Collapse)
  NODE(AccClause, Copy)
  NODE(AccClause, Copyin)
  NODE(AccClause, Copyout)
  NODE(AccClause, Create)
  NODE(AccClause, Default)
  NODE(AccClause, DefaultAsync)
  NODE(AccClause, Delete)
  NODE(AccClause, Detach)
  NODE(AccClause, Device)
  NODE(AccClause, DeviceNum)
  NODE(AccClause, DevicePtr)
  NODE(AccClause, DeviceResident)
  NODE(AccClause, DeviceType)
  NODE(AccClause, Finalize)
  NODE(AccClause, FirstPrivate)
  NODE(AccClause, Gang)
  NODE(AccClause, Host)
  NODE(AccClause, If)
  NODE(AccClause, IfPresent)
  NODE(AccClause, Independent)
  NODE(AccClause, Link)
  NODE(AccClause, NoCreate)
  NODE(AccClause, NoHost)
  NODE(AccClause, NumGangs)
  NODE(AccClause, NumWorkers)
  NODE(AccClause, Present)
  NODE(AccClause, Private)
  NODE(AccClause, Tile)
  NODE(AccClause, UseDevice)
  NODE(AccClause, Read)
  NODE(AccClause, Reduction)
  NODE(AccClause, Self)
  NODE(AccClause, Seq)
  NODE(AccClause, Vector)
  NODE(AccClause, VectorLength)
  NODE(AccClause, Wait)
  NODE(AccClause, Worker)
  NODE(AccClause, Write)
  NODE(AccClause, Unknown)
  NODE(parser, AccDefaultClause)
  NODE_ENUM(parser::AccDefaultClause, Arg)
  NODE(parser, AccClauseList)
  NODE(parser, AccCombinedDirective)
  NODE(parser, AccDataModifier)
  NODE_ENUM(parser::AccDataModifier, Modifier)
  NODE(parser, AccDeclarativeDirective)
  NODE(parser, AccEndAtomic)
  NODE(parser, AccEndBlockDirective)
  NODE(parser, AccEndCombinedDirective)
  NODE(parser, AccGangArgument)
  NODE(parser, AccObject)
  NODE(parser, AccObjectList)
  NODE(parser, AccObjectListWithModifier)
  NODE(parser, AccObjectListWithReduction)
  NODE(parser, AccReductionOperator)
  NODE(parser, AccSizeExpr)
  NODE(parser, AccSizeExprList)
  NODE(parser, AccStandaloneDirective)
  NODE(parser, AccLoopDirective)
  NODE(parser, AccWaitArgument)
  static std::string GetNodeName(const llvm::acc::Directive &x) {
    return llvm::Twine(
        "llvm::acc::Directive = ", llvm::acc::getOpenACCDirectiveName(x))
        .str();
  }
  NODE(parser, AcImpliedDo)
  NODE(parser, AcImpliedDoControl)
  NODE(parser, AcValue)
  NODE(parser, AccessStmt)
  NODE(parser, AccessId)
  NODE(parser, AccessSpec)
  NODE_ENUM(AccessSpec, Kind)
  NODE(parser, AcSpec)
  NODE(parser, ActionStmt)
  NODE(parser, ActualArg)
  NODE(ActualArg, PercentRef)
  NODE(ActualArg, PercentVal)
  NODE(parser, ActualArgSpec)
  NODE(AcValue, Triplet)
  NODE(parser, AllocOpt)
  NODE(AllocOpt, Mold)
  NODE(AllocOpt, Source)
  NODE(parser, Allocatable)
  NODE(parser, AllocatableStmt)
  NODE(parser, AllocateCoarraySpec)
  NODE(parser, AllocateObject)
  NODE(parser, AllocateShapeSpec)
  NODE(parser, AllocateStmt)
  NODE(parser, Allocation)
  NODE(parser, AltReturnSpec)
  NODE(parser, ArithmeticIfStmt)
  NODE(parser, ArrayConstructor)
  NODE(parser, ArrayElement)
  NODE(parser, ArraySpec)
  NODE(parser, AssignStmt)
  NODE(parser, AssignedGotoStmt)
  NODE(parser, AssignmentStmt)
  NODE(parser, AssociateConstruct)
  NODE(parser, AssociateStmt)
  NODE(parser, Association)
  NODE(parser, AssumedImpliedSpec)
  NODE(parser, AssumedRankSpec)
  NODE(parser, AssumedShapeSpec)
  NODE(parser, AssumedSizeSpec)
  NODE(parser, Asynchronous)
  NODE(parser, AsynchronousStmt)
  NODE(parser, AttrSpec)
  NODE(parser, BOZLiteralConstant)
  NODE(parser, BackspaceStmt)
  NODE(parser, BasedPointer)
  NODE(parser, BasedPointerStmt)
  NODE(parser, BindAttr)
  NODE(BindAttr, Deferred)
  NODE(BindAttr, Non_Overridable)
  NODE(parser, BindEntity)
  NODE_ENUM(BindEntity, Kind)
  NODE(parser, BindStmt)
  NODE(parser, Block)
  NODE(parser, BlockConstruct)
  NODE(parser, BlockData)
  NODE(parser, BlockDataStmt)
  NODE(parser, BlockSpecificationPart)
  NODE(parser, BlockStmt)
  NODE(parser, BoundsRemapping)
  NODE(parser, BoundsSpec)
  NODE(parser, Call)
  NODE(parser, CallStmt)
  NODE(parser, CaseConstruct)
  NODE(CaseConstruct, Case)
  NODE(parser, CaseSelector)
  NODE(parser, CaseStmt)
  NODE(parser, CaseValueRange)
  NODE(CaseValueRange, Range)
  NODE(parser, ChangeTeamConstruct)
  NODE(parser, ChangeTeamStmt)
  NODE(parser, CharLength)
  NODE(parser, CharLiteralConstant)
  NODE(parser, CharLiteralConstantSubstring)
  NODE(parser, CharSelector)
  NODE(CharSelector, LengthAndKind)
  NODE(parser, CloseStmt)
  NODE(CloseStmt, CloseSpec)
  NODE(parser, CoarrayAssociation)
  NODE(parser, CoarraySpec)
  NODE(parser, CodimensionDecl)
  NODE(parser, CodimensionStmt)
  NODE(parser, CoindexedNamedObject)
  NODE(parser, CommonBlockObject)
  NODE(parser, CommonStmt)
  NODE(CommonStmt, Block)
  NODE(parser, CompilerDirective)
  NODE(CompilerDirective, IgnoreTKR)
  NODE(CompilerDirective, NameValue)
  NODE(parser, ComplexLiteralConstant)
  NODE(parser, ComplexPart)
  NODE(parser, ComponentArraySpec)
  NODE(parser, ComponentAttrSpec)
  NODE(parser, ComponentDataSource)
  NODE(parser, ComponentDecl)
  NODE(parser, ComponentDefStmt)
  NODE(parser, ComponentSpec)
  NODE(parser, ComputedGotoStmt)
  NODE(parser, ConcurrentControl)
  NODE(parser, ConcurrentHeader)
  NODE(parser, ConnectSpec)
  NODE(ConnectSpec, CharExpr)
  NODE_ENUM(ConnectSpec::CharExpr, Kind)
  NODE(ConnectSpec, Newunit)
  NODE(ConnectSpec, Recl)
  NODE(parser, ConstantValue)
  NODE(parser, ContainsStmt)
  NODE(parser, Contiguous)
  NODE(parser, ContiguousStmt)
  NODE(parser, ContinueStmt)
  NODE(parser, CriticalConstruct)
  NODE(parser, CriticalStmt)
  NODE(parser, CycleStmt)
  NODE(parser, DataComponentDefStmt)
  NODE(parser, DataIDoObject)
  NODE(parser, DataImpliedDo)
  NODE(parser, DataRef)
  NODE(parser, DataStmt)
  NODE(parser, DataStmtConstant)
  NODE(parser, DataStmtObject)
  NODE(parser, DataStmtRepeat)
  NODE(parser, DataStmtSet)
  NODE(parser, DataStmtValue)
  NODE(parser, DeallocateStmt)
  NODE(parser, DeclarationConstruct)
  NODE(parser, DeclarationTypeSpec)
  NODE(DeclarationTypeSpec, Class)
  NODE(DeclarationTypeSpec, ClassStar)
  NODE(DeclarationTypeSpec, Record)
  NODE(DeclarationTypeSpec, Type)
  NODE(DeclarationTypeSpec, TypeStar)
  NODE(parser, Default)
  NODE(parser, DeferredCoshapeSpecList)
  NODE(parser, DeferredShapeSpecList)
  NODE(parser, DefinedOpName)
  NODE(parser, DefinedOperator)
  NODE_ENUM(DefinedOperator, IntrinsicOperator)
  NODE(parser, DerivedTypeDef)
  NODE(parser, DerivedTypeSpec)
  NODE(parser, DerivedTypeStmt)
  NODE(parser, Designator)
  NODE(parser, DimensionStmt)
  NODE(DimensionStmt, Declaration)
  NODE(parser, DoConstruct)
  NODE(parser, DummyArg)
  NODE(parser, ElseIfStmt)
  NODE(parser, ElseStmt)
  NODE(parser, ElsewhereStmt)
  NODE(parser, EndAssociateStmt)
  NODE(parser, EndBlockDataStmt)
  NODE(parser, EndBlockStmt)
  NODE(parser, EndChangeTeamStmt)
  NODE(parser, EndCriticalStmt)
  NODE(parser, EndDoStmt)
  NODE(parser, EndEnumStmt)
  NODE(parser, EndForallStmt)
  NODE(parser, EndFunctionStmt)
  NODE(parser, EndIfStmt)
  NODE(parser, EndInterfaceStmt)
  NODE(parser, EndLabel)
  NODE(parser, EndModuleStmt)
  NODE(parser, EndMpSubprogramStmt)
  NODE(parser, EndProgramStmt)
  NODE(parser, EndSelectStmt)
  NODE(parser, EndSubmoduleStmt)
  NODE(parser, EndSubroutineStmt)
  NODE(parser, EndTypeStmt)
  NODE(parser, EndWhereStmt)
  NODE(parser, EndfileStmt)
  NODE(parser, EntityDecl)
  NODE(parser, EntryStmt)
  NODE(parser, EnumDef)
  NODE(parser, EnumDefStmt)
  NODE(parser, Enumerator)
  NODE(parser, EnumeratorDefStmt)
  NODE(parser, EorLabel)
  NODE(parser, EquivalenceObject)
  NODE(parser, EquivalenceStmt)
  NODE(parser, ErrLabel)
  NODE(parser, ErrorRecovery)
  NODE(parser, EventPostStmt)
  NODE(parser, EventWaitStmt)
  NODE(EventWaitStmt, EventWaitSpec)
  NODE(parser, ExecutableConstruct)
  NODE(parser, ExecutionPart)
  NODE(parser, ExecutionPartConstruct)
  NODE(parser, ExitStmt)
  NODE(parser, ExplicitCoshapeSpec)
  NODE(parser, ExplicitShapeSpec)
  NODE(parser, Expr)
  NODE(Expr, Parentheses)
  NODE(Expr, UnaryPlus)
  NODE(Expr, Negate)
  NODE(Expr, NOT)
  NODE(Expr, PercentLoc)
  NODE(Expr, DefinedUnary)
  NODE(Expr, Power)
  NODE(Expr, Multiply)
  NODE(Expr, Divide)
  NODE(Expr, Add)
  NODE(Expr, Subtract)
  NODE(Expr, Concat)
  NODE(Expr, LT)
  NODE(Expr, LE)
  NODE(Expr, EQ)
  NODE(Expr, NE)
  NODE(Expr, GE)
  NODE(Expr, GT)
  NODE(Expr, AND)
  NODE(Expr, OR)
  NODE(Expr, EQV)
  NODE(Expr, NEQV)
  NODE(Expr, DefinedBinary)
  NODE(Expr, ComplexConstructor)
  NODE(parser, External)
  NODE(parser, ExternalStmt)
  NODE(parser, FailImageStmt)
  NODE(parser, FileUnitNumber)
  NODE(parser, FinalProcedureStmt)
  NODE(parser, FlushStmt)
  NODE(parser, ForallAssignmentStmt)
  NODE(parser, ForallBodyConstruct)
  NODE(parser, ForallConstruct)
  NODE(parser, ForallConstructStmt)
  NODE(parser, ForallStmt)
  NODE(parser, FormTeamStmt)
  NODE(FormTeamStmt, FormTeamSpec)
  NODE(parser, Format)
  NODE(parser, FormatStmt)
  NODE(parser, FunctionReference)
  NODE(parser, FunctionStmt)
  NODE(parser, FunctionSubprogram)
  NODE(parser, GenericSpec)
  NODE(GenericSpec, Assignment)
  NODE(GenericSpec, ReadFormatted)
  NODE(GenericSpec, ReadUnformatted)
  NODE(GenericSpec, WriteFormatted)
  NODE(GenericSpec, WriteUnformatted)
  NODE(parser, GenericStmt)
  NODE(parser, GotoStmt)
  NODE(parser, HollerithLiteralConstant)
  NODE(parser, IdExpr)
  NODE(parser, IdVariable)
  NODE(parser, IfConstruct)
  NODE(IfConstruct, ElseBlock)
  NODE(IfConstruct, ElseIfBlock)
  NODE(parser, IfStmt)
  NODE(parser, IfThenStmt)
  NODE(parser, TeamValue)
  NODE(parser, ImageSelector)
  NODE(parser, ImageSelectorSpec)
  NODE(ImageSelectorSpec, Stat)
  NODE(ImageSelectorSpec, Team_Number)
  NODE(parser, ImplicitPart)
  NODE(parser, ImplicitPartStmt)
  NODE(parser, ImplicitSpec)
  NODE(parser, ImplicitStmt)
  NODE_ENUM(ImplicitStmt, ImplicitNoneNameSpec)
  NODE(parser, ImpliedShapeSpec)
  NODE(parser, ImportStmt)
  NODE(parser, Initialization)
  NODE(parser, InputImpliedDo)
  NODE(parser, InputItem)
  NODE(parser, InquireSpec)
  NODE(InquireSpec, CharVar)
  NODE_ENUM(InquireSpec::CharVar, Kind)
  NODE(InquireSpec, IntVar)
  NODE_ENUM(InquireSpec::IntVar, Kind)
  NODE(InquireSpec, LogVar)
  NODE_ENUM(InquireSpec::LogVar, Kind)
  NODE(parser, InquireStmt)
  NODE(InquireStmt, Iolength)
  NODE(parser, IntegerTypeSpec)
  NODE(parser, IntentSpec)
  NODE_ENUM(IntentSpec, Intent)
  NODE(parser, IntentStmt)
  NODE(parser, InterfaceBlock)
  NODE(parser, InterfaceBody)
  NODE(InterfaceBody, Function)
  NODE(InterfaceBody, Subroutine)
  NODE(parser, InterfaceSpecification)
  NODE(parser, InterfaceStmt)
  NODE(parser, InternalSubprogram)
  NODE(parser, InternalSubprogramPart)
  NODE(parser, Intrinsic)
  NODE(parser, IntrinsicStmt)
  NODE(parser, IntrinsicTypeSpec)
  NODE(IntrinsicTypeSpec, Character)
  NODE(IntrinsicTypeSpec, Complex)
  NODE(IntrinsicTypeSpec, DoubleComplex)
  NODE(IntrinsicTypeSpec, DoublePrecision)
  NODE(IntrinsicTypeSpec, Logical)
  NODE(IntrinsicTypeSpec, Real)
  NODE(parser, IoControlSpec)
  NODE(IoControlSpec, Asynchronous)
  NODE(IoControlSpec, CharExpr)
  NODE_ENUM(IoControlSpec::CharExpr, Kind)
  NODE(IoControlSpec, Pos)
  NODE(IoControlSpec, Rec)
  NODE(IoControlSpec, Size)
  NODE(parser, IoUnit)
  NODE(parser, Keyword)
  NODE(parser, KindParam)
  NODE(parser, KindSelector)
  NODE(KindSelector, StarSize)
  NODE(parser, LabelDoStmt)
  NODE(parser, LanguageBindingSpec)
  NODE(parser, LengthSelector)
  NODE(parser, LetterSpec)
  NODE(parser, LiteralConstant)
  NODE(parser, IntLiteralConstant)
  NODE(parser, LocalitySpec)
  NODE(LocalitySpec, DefaultNone)
  NODE(LocalitySpec, Local)
  NODE(LocalitySpec, LocalInit)
  NODE(LocalitySpec, Shared)
  NODE(parser, LockStmt)
  NODE(LockStmt, LockStat)
  NODE(parser, LogicalLiteralConstant)
  NODE_NAME(LoopControl::Bounds, "LoopBounds")
  NODE_NAME(AcImpliedDoControl::Bounds, "LoopBounds")
  NODE_NAME(DataImpliedDo::Bounds, "LoopBounds")
  NODE(parser, LoopControl)
  NODE(LoopControl, Concurrent)
  NODE(parser, MainProgram)
  NODE(parser, Map)
  NODE(Map, EndMapStmt)
  NODE(Map, MapStmt)
  NODE(parser, MaskedElsewhereStmt)
  NODE(parser, Module)
  NODE(parser, ModuleStmt)
  NODE(parser, ModuleSubprogram)
  NODE(parser, ModuleSubprogramPart)
  NODE(parser, MpSubprogramStmt)
  NODE(parser, MsgVariable)
  NODE(parser, Name)
  NODE(parser, NamedConstant)
  NODE(parser, NamedConstantDef)
  NODE(parser, NamelistStmt)
  NODE(NamelistStmt, Group)
  NODE(parser, NonLabelDoStmt)
  NODE(parser, NoPass)
  NODE(parser, NullifyStmt)
  NODE(parser, NullInit)
  NODE(parser, ObjectDecl)
  NODE(parser, OldParameterStmt)
  NODE(parser, OmpAlignedClause)
  NODE(parser, OmpAtomic)
  NODE(parser, OmpAtomicCapture)
  NODE(OmpAtomicCapture, Stmt1)
  NODE(OmpAtomicCapture, Stmt2)
  NODE(parser, OmpAtomicRead)
  NODE(parser, OmpAtomicUpdate)
  NODE(parser, OmpAtomicWrite)
  NODE(parser, OmpBeginBlockDirective)
  NODE(parser, OmpBeginLoopDirective)
  NODE(parser, OmpBeginSectionsDirective)
  NODE(parser, OmpBlockDirective)
  static std::string GetNodeName(const llvm::omp::Directive &x) {
    return llvm::Twine(
        "llvm::omp::Directive = ", llvm::omp::getOpenMPDirectiveName(x))
        .str();
  }
  NODE(parser, OmpCancelType)
  NODE_ENUM(OmpCancelType, Type)
  NODE(parser, OmpClause)
  NODE(parser, OmpClauseList)
  NODE(OmpClause, Collapse)
  NODE(OmpClause, Copyin)
  NODE(OmpClause, Copyprivate)
  NODE(OmpClause, Device)
  NODE(OmpClause, DistSchedule)
  NODE(OmpClause, Final)
  NODE(OmpClause, Firstprivate)
  NODE(OmpClause, From)
  NODE(OmpClause, Grainsize)
  NODE(OmpClause, Inbranch)
  NODE(OmpClause, Lastprivate)
  NODE(OmpClause, Mergeable)
  NODE(OmpClause, Nogroup)
  NODE(OmpClause, Notinbranch)
  NODE(OmpClause, Threads)
  NODE(OmpClause, Simd)
  NODE(OmpClause, NumTasks)
  NODE(OmpClause, NumTeams)
  NODE(OmpClause, NumThreads)
  NODE(OmpClause, Ordered)
  NODE(OmpClause, Priority)
  NODE(OmpClause, Private)
  NODE(OmpClause, Safelen)
  NODE(OmpClause, Shared)
  NODE(OmpClause, Simdlen)
  NODE(OmpClause, ThreadLimit)
  NODE(OmpClause, To)
  NODE(OmpClause, Link)
  NODE(OmpClause, Uniform)
  NODE(OmpClause, Untied)
  NODE(OmpClause, UseDevicePtr)
  NODE(OmpClause, IsDevicePtr)
  NODE(parser, OmpCriticalDirective)
  NODE(OmpCriticalDirective, Hint)
  NODE(parser, OmpDeclareTargetSpecifier)
  NODE(parser, OmpDeclareTargetWithClause)
  NODE(parser, OmpDeclareTargetWithList)
  NODE(parser, OmpDefaultClause)
  NODE_ENUM(OmpDefaultClause, Type)
  NODE(parser, OmpDefaultmapClause)
  NODE_ENUM(OmpDefaultmapClause, ImplicitBehavior)
  NODE_ENUM(OmpDefaultmapClause, VariableCategory)
  NODE(parser, OmpDependClause)
  NODE(OmpDependClause, InOut)
  NODE(OmpDependClause, Sink)
  NODE(OmpDependClause, Source)
  NODE(parser, OmpDependenceType)
  NODE_ENUM(OmpDependenceType, Type)
  NODE(parser, OmpDependSinkVec)
  NODE(parser, OmpDependSinkVecLength)
  NODE(parser, OmpEndAtomic)
  NODE(parser, OmpEndBlockDirective)
  NODE(parser, OmpEndCriticalDirective)
  NODE(parser, OmpEndLoopDirective)
  NODE(parser, OmpEndSectionsDirective)
  NODE(parser, OmpIfClause)
  NODE_ENUM(OmpIfClause, DirectiveNameModifier)
  NODE(parser, OmpLinearClause)
  NODE(OmpLinearClause, WithModifier)
  NODE(OmpLinearClause, WithoutModifier)
  NODE(parser, OmpLinearModifier)
  NODE_ENUM(OmpLinearModifier, Type)
  NODE(parser, OmpLoopDirective)
  NODE(parser, OmpMapClause)
  NODE(parser, OmpMapType)
  NODE(OmpMapType, Always)
  NODE_ENUM(OmpMapType, Type)
  NODE(parser, OmpMemoryClause)
  NODE_ENUM(OmpMemoryClause, MemoryOrder)
  NODE(parser, OmpMemoryClauseList)
  NODE(parser, OmpMemoryClausePostList)
  NODE(parser, OmpNowait)
  NODE(parser, OmpObject)
  NODE(parser, OmpObjectList)
  NODE(parser, OmpProcBindClause)
  NODE_ENUM(OmpProcBindClause, Type)
  NODE(parser, OmpReductionClause)
  NODE(parser, OmpReductionCombiner)
  NODE(OmpReductionCombiner, FunctionCombiner)
  NODE(parser, OmpReductionInitializerClause)
  NODE(parser, OmpReductionOperator)
  NODE(parser, OmpScheduleClause)
  NODE_ENUM(OmpScheduleClause, ScheduleType)
  NODE(parser, OmpScheduleModifier)
  NODE(OmpScheduleModifier, Modifier1)
  NODE(OmpScheduleModifier, Modifier2)
  NODE(parser, OmpScheduleModifierType)
  NODE_ENUM(OmpScheduleModifierType, ModType)
  NODE(parser, OmpSectionBlocks)
  NODE(parser, OmpSectionsDirective)
  NODE(parser, OmpSimpleStandaloneDirective)
  NODE(parser, Only)
  NODE(parser, OpenACCAtomicConstruct)
  NODE(parser, OpenACCBlockConstruct)
  NODE(parser, OpenACCCacheConstruct)
  NODE(parser, OpenACCCombinedConstruct)
  NODE(parser, OpenACCConstruct)
  NODE(parser, OpenACCDeclarativeConstruct)
  NODE(parser, OpenACCLoopConstruct)
  NODE(parser, OpenACCRoutineConstruct)
  NODE(parser, OpenACCStandaloneDeclarativeConstruct)
  NODE(parser, OpenACCStandaloneConstruct)
  NODE(parser, OpenACCWaitConstruct)
  NODE(parser, OpenMPAtomicConstruct)
  NODE(parser, OpenMPBlockConstruct)
  NODE(parser, OpenMPCancelConstruct)
  NODE(OpenMPCancelConstruct, If)
  NODE(parser, OpenMPCancellationPointConstruct)
  NODE(parser, OpenMPConstruct)
  NODE(parser, OpenMPCriticalConstruct)
  NODE(parser, OpenMPDeclarativeConstruct)
  NODE(parser, OpenMPDeclareReductionConstruct)
  NODE(parser, OpenMPDeclareSimdConstruct)
  NODE(parser, OpenMPDeclareTargetConstruct)
  NODE(parser, OmpFlushMemoryClause)
  NODE_ENUM(OmpFlushMemoryClause, FlushMemoryOrder)
  NODE(parser, OpenMPFlushConstruct)
  NODE(parser, OpenMPLoopConstruct)
  NODE(parser, OpenMPSimpleStandaloneConstruct)
  NODE(parser, OpenMPStandaloneConstruct)
  NODE(parser, OpenMPSectionsConstruct)
  NODE(parser, OpenMPThreadprivate)
  NODE(parser, OpenStmt)
  NODE(parser, Optional)
  NODE(parser, OptionalStmt)
  NODE(parser, OtherSpecificationStmt)
  NODE(parser, OutputImpliedDo)
  NODE(parser, OutputItem)
  NODE(parser, Parameter)
  NODE(parser, ParameterStmt)
  NODE(parser, ParentIdentifier)
  NODE(parser, Pass)
  NODE(parser, PauseStmt)
  NODE(parser, Pointer)
  NODE(parser, PointerAssignmentStmt)
  NODE(PointerAssignmentStmt, Bounds)
  NODE(parser, PointerDecl)
  NODE(parser, PointerObject)
  NODE(parser, PointerStmt)
  NODE(parser, PositionOrFlushSpec)
  NODE(parser, PrefixSpec)
  NODE(PrefixSpec, Elemental)
  NODE(PrefixSpec, Impure)
  NODE(PrefixSpec, Module)
  NODE(PrefixSpec, Non_Recursive)
  NODE(PrefixSpec, Pure)
  NODE(PrefixSpec, Recursive)
  NODE(parser, PrintStmt)
  NODE(parser, PrivateStmt)
  NODE(parser, PrivateOrSequence)
  NODE(parser, ProcAttrSpec)
  NODE(parser, ProcComponentAttrSpec)
  NODE(parser, ProcComponentDefStmt)
  NODE(parser, ProcComponentRef)
  NODE(parser, ProcDecl)
  NODE(parser, ProcInterface)
  NODE(parser, ProcPointerInit)
  NODE(parser, ProcedureDeclarationStmt)
  NODE(parser, ProcedureDesignator)
  NODE(parser, ProcedureStmt)
  NODE_ENUM(ProcedureStmt, Kind)
  NODE(parser, Program)
  NODE(parser, ProgramStmt)
  NODE(parser, ProgramUnit)
  NODE(parser, Protected)
  NODE(parser, ProtectedStmt)
  NODE(parser, ReadStmt)
  NODE(parser, RealLiteralConstant)
  NODE(RealLiteralConstant, Real)
  NODE(parser, Rename)
  NODE(Rename, Names)
  NODE(Rename, Operators)
  NODE(parser, ReturnStmt)
  NODE(parser, RewindStmt)
  NODE(parser, Save)
  NODE(parser, SaveStmt)
  NODE(parser, SavedEntity)
  NODE_ENUM(SavedEntity, Kind)
  NODE(parser, SectionSubscript)
  NODE(parser, SelectCaseStmt)
  NODE(parser, SelectRankCaseStmt)
  NODE(SelectRankCaseStmt, Rank)
  NODE(parser, SelectRankConstruct)
  NODE(SelectRankConstruct, RankCase)
  NODE(parser, SelectRankStmt)
  NODE(parser, SelectTypeConstruct)
  NODE(SelectTypeConstruct, TypeCase)
  NODE(parser, SelectTypeStmt)
  NODE(parser, Selector)
  NODE(parser, SeparateModuleSubprogram)
  NODE(parser, SequenceStmt)
  NODE(parser, Sign)
  NODE(parser, SignedComplexLiteralConstant)
  NODE(parser, SignedIntLiteralConstant)
  NODE(parser, SignedRealLiteralConstant)
  NODE(parser, SpecificationConstruct)
  NODE(parser, SpecificationExpr)
  NODE(parser, SpecificationPart)
  NODE(parser, Star)
  NODE(parser, StatOrErrmsg)
  NODE(parser, StatVariable)
  NODE(parser, StatusExpr)
  NODE(parser, StmtFunctionStmt)
  NODE(parser, StopCode)
  NODE(parser, StopStmt)
  NODE_ENUM(StopStmt, Kind)
  NODE(parser, StructureComponent)
  NODE(parser, StructureConstructor)
  NODE(parser, StructureDef)
  NODE(StructureDef, EndStructureStmt)
  NODE(parser, StructureField)
  NODE(parser, StructureStmt)
  NODE(parser, Submodule)
  NODE(parser, SubmoduleStmt)
  NODE(parser, SubroutineStmt)
  NODE(parser, SubroutineSubprogram)
  NODE(parser, SubscriptTriplet)
  NODE(parser, Substring)
  NODE(parser, SubstringRange)
  NODE(parser, Suffix)
  NODE(parser, SyncAllStmt)
  NODE(parser, SyncImagesStmt)
  NODE(SyncImagesStmt, ImageSet)
  NODE(parser, SyncMemoryStmt)
  NODE(parser, SyncTeamStmt)
  NODE(parser, Target)
  NODE(parser, TargetStmt)
  NODE(parser, TypeAttrSpec)
  NODE(TypeAttrSpec, BindC)
  NODE(TypeAttrSpec, Extends)
  NODE(parser, TypeBoundGenericStmt)
  NODE(parser, TypeBoundProcBinding)
  NODE(parser, TypeBoundProcDecl)
  NODE(parser, TypeBoundProcedurePart)
  NODE(parser, TypeBoundProcedureStmt)
  NODE(TypeBoundProcedureStmt, WithInterface)
  NODE(TypeBoundProcedureStmt, WithoutInterface)
  NODE(parser, TypeDeclarationStmt)
  NODE(parser, TypeGuardStmt)
  NODE(TypeGuardStmt, Guard)
  NODE(parser, TypeParamDecl)
  NODE(parser, TypeParamDefStmt)
  NODE(common, TypeParamAttr)
  NODE(parser, TypeParamSpec)
  NODE(parser, TypeParamValue)
  NODE(TypeParamValue, Deferred)
  NODE(parser, TypeSpec)
  NODE(parser, Union)
  NODE(Union, EndUnionStmt)
  NODE(Union, UnionStmt)
  NODE(parser, UnlockStmt)
  NODE(parser, UseStmt)
  NODE_ENUM(UseStmt, ModuleNature)
  NODE(parser, Value)
  NODE(parser, ValueStmt)
  NODE(parser, Variable)
  NODE(parser, Verbatim)
  NODE(parser, Volatile)
  NODE(parser, VolatileStmt)
  NODE(parser, WaitSpec)
  NODE(parser, WaitStmt)
  NODE(parser, WhereBodyConstruct)
  NODE(parser, WhereConstruct)
  NODE(WhereConstruct, Elsewhere)
  NODE(WhereConstruct, MaskedElsewhere)
  NODE(parser, WhereConstructStmt)
  NODE(parser, WhereStmt)
  NODE(parser, WriteStmt)
#undef NODE
#undef NODE_NAME

  template <typename T> bool Pre(const T &x) {
    std::string fortran{AsFortran<T>(x)};
    if (fortran.empty() && (UnionTrait<T> || WrapperTrait<T>)) {
      Prefix(GetNodeName(x));
    } else {
      IndentEmptyLine();
      out_ << GetNodeName(x);
      if (!fortran.empty()) {
        out_ << " = '" << fortran << '\'';
      }
      EndLine();
      ++indent_;
    }
    return true;
  }

  template <typename T> void Post(const T &x) {
    if (AsFortran<T>(x).empty() && (UnionTrait<T> || WrapperTrait<T>)) {
      EndLineIfNonempty();
    } else {
      --indent_;
    }
  }

  // A few types we want to ignore

  bool Pre(const CharBlock &) { return true; }
  void Post(const CharBlock &) {}

  template <typename T> bool Pre(const Statement<T> &) { return true; }
  template <typename T> void Post(const Statement<T> &) {}
  template <typename T> bool Pre(const UnlabeledStatement<T> &) { return true; }
  template <typename T> void Post(const UnlabeledStatement<T> &) {}

  template <typename T> bool Pre(const common::Indirection<T> &) {
    return true;
  }
  template <typename T> void Post(const common::Indirection<T> &) {}

  template <typename A> bool Pre(const Scalar<A> &) {
    Prefix("Scalar");
    return true;
  }
  template <typename A> void Post(const Scalar<A> &) { EndLineIfNonempty(); }

  template <typename A> bool Pre(const Constant<A> &) {
    Prefix("Constant");
    return true;
  }
  template <typename A> void Post(const Constant<A> &) { EndLineIfNonempty(); }

  template <typename A> bool Pre(const Integer<A> &) {
    Prefix("Integer");
    return true;
  }
  template <typename A> void Post(const Integer<A> &) { EndLineIfNonempty(); }

  template <typename A> bool Pre(const Logical<A> &) {
    Prefix("Logical");
    return true;
  }
  template <typename A> void Post(const Logical<A> &) { EndLineIfNonempty(); }

  template <typename A> bool Pre(const DefaultChar<A> &) {
    Prefix("DefaultChar");
    return true;
  }
  template <typename A> void Post(const DefaultChar<A> &) {
    EndLineIfNonempty();
  }

  template <typename... A> bool Pre(const std::tuple<A...> &) { return true; }
  template <typename... A> void Post(const std::tuple<A...> &) {}

  template <typename... A> bool Pre(const std::variant<A...> &) { return true; }
  template <typename... A> void Post(const std::variant<A...> &) {}

protected:
  // Return a Fortran representation of this node to include in the dump
  template <typename T> std::string AsFortran(const T &x) {
    std::string buf;
    llvm::raw_string_ostream ss{buf};
    if constexpr (std::is_same_v<T, Expr>) {
      if (asFortran_ && x.typedExpr) {
        asFortran_->expr(ss, *x.typedExpr);
      }
    } else if constexpr (std::is_same_v<T, AssignmentStmt> ||
        std::is_same_v<T, PointerAssignmentStmt>) {
      if (asFortran_ && x.typedAssignment) {
        asFortran_->assignment(ss, *x.typedAssignment);
      }
    } else if constexpr (std::is_same_v<T, CallStmt>) {
      if (asFortran_ && x.typedCall) {
        asFortran_->call(ss, *x.typedCall);
      }
    } else if constexpr (std::is_same_v<T, IntLiteralConstant> ||
        std::is_same_v<T, SignedIntLiteralConstant>) {
      ss << std::get<CharBlock>(x.t);
    } else if constexpr (std::is_same_v<T, RealLiteralConstant::Real>) {
      ss << x.source;
    } else if constexpr (std::is_same_v<T, std::string> ||
        std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>) {
      ss << x;
    }
    if (ss.tell()) {
      return ss.str();
    }
    if constexpr (std::is_same_v<T, Name>) {
      return x.source.ToString();
#ifdef SHOW_ALL_SOURCE_MEMBERS
    } else if constexpr (HasSource<T>::value) {
      return x.source.ToString();
#endif
    } else if constexpr (std::is_same_v<T, std::string>) {
      return x;
    } else {
      return "";
    }
  }

  void IndentEmptyLine() {
    if (emptyline_ && indent_ > 0) {
      for (int i{0}; i < indent_; ++i) {
        out_ << "| ";
      }
      emptyline_ = false;
    }
  }

  void Prefix(const char *str) {
    IndentEmptyLine();
    out_ << str << " -> ";
    emptyline_ = false;
  }

  void Prefix(const std::string &str) {
    IndentEmptyLine();
    out_ << str << " -> ";
    emptyline_ = false;
  }

  void EndLine() {
    out_ << '\n';
    emptyline_ = true;
  }

  void EndLineIfNonempty() {
    if (!emptyline_) {
      EndLine();
    }
  }

private:
  int indent_{0};
  llvm::raw_ostream &out_;
  const AnalyzedObjectsAsFortran *const asFortran_;
  bool emptyline_{false};
};

template <typename T>
llvm::raw_ostream &DumpTree(llvm::raw_ostream &out, const T &x,
    const AnalyzedObjectsAsFortran *asFortran = nullptr) {
  ParseTreeDumper dumper{out, asFortran};
  Walk(x, dumper);
  return out;
}

} // namespace Fortran::parser
#endif // FORTRAN_PARSER_DUMP_PARSE_TREE_H_
