//===-- IO.cpp -- IO statement lowering -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/IO.h"
#include "flang/Common/uint128.h"
#include "flang/Evaluate/tools.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/VectorSubscripts.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/io-api.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-io"

// Define additional runtime type models specific to IO.
namespace fir::runtime {
template <>
constexpr TypeBuilderFunc getModel<Fortran::runtime::io::IoStatementState *>() {
  return getModel<char *>();
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::runtime::io::Iostat>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context,
                                  8 * sizeof(Fortran::runtime::io::Iostat));
  };
}
template <>
constexpr TypeBuilderFunc
getModel<const Fortran::runtime::io::NamelistGroup &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(mlir::TupleType::get(context));
  };
}
template <>
constexpr TypeBuilderFunc
getModel<const Fortran::runtime::io::NonTbpDefinedIoTable *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(mlir::TupleType::get(context));
  };
}
} // namespace fir::runtime

using namespace Fortran::runtime::io;

#define mkIOKey(X) FirmkKey(IONAME(X))

namespace Fortran::lower {
/// Static table of IO runtime calls
///
/// This logical map contains the name and type builder function for each IO
/// runtime function listed in the tuple. This table is fully constructed at
/// compile-time. Use the `mkIOKey` macro to access the table.
static constexpr std::tuple<
    mkIOKey(BeginBackspace), mkIOKey(BeginClose), mkIOKey(BeginEndfile),
    mkIOKey(BeginExternalFormattedInput), mkIOKey(BeginExternalFormattedOutput),
    mkIOKey(BeginExternalListInput), mkIOKey(BeginExternalListOutput),
    mkIOKey(BeginFlush), mkIOKey(BeginInquireFile),
    mkIOKey(BeginInquireIoLength), mkIOKey(BeginInquireUnit),
    mkIOKey(BeginInternalArrayFormattedInput),
    mkIOKey(BeginInternalArrayFormattedOutput),
    mkIOKey(BeginInternalArrayListInput), mkIOKey(BeginInternalArrayListOutput),
    mkIOKey(BeginInternalFormattedInput), mkIOKey(BeginInternalFormattedOutput),
    mkIOKey(BeginInternalListInput), mkIOKey(BeginInternalListOutput),
    mkIOKey(BeginOpenNewUnit), mkIOKey(BeginOpenUnit), mkIOKey(BeginRewind),
    mkIOKey(BeginUnformattedInput), mkIOKey(BeginUnformattedOutput),
    mkIOKey(BeginWait), mkIOKey(BeginWaitAll),
    mkIOKey(CheckUnitNumberInRange64), mkIOKey(CheckUnitNumberInRange128),
    mkIOKey(EnableHandlers), mkIOKey(EndIoStatement), mkIOKey(GetIoLength),
    mkIOKey(GetIoMsg), mkIOKey(GetNewUnit), mkIOKey(GetSize),
    mkIOKey(InputAscii), mkIOKey(InputComplex32), mkIOKey(InputComplex64),
    mkIOKey(InputDerivedType), mkIOKey(InputDescriptor), mkIOKey(InputInteger),
    mkIOKey(InputLogical), mkIOKey(InputNamelist), mkIOKey(InputReal32),
    mkIOKey(InputReal64), mkIOKey(InputUnformattedBlock),
    mkIOKey(InquireCharacter), mkIOKey(InquireInteger64),
    mkIOKey(InquireLogical), mkIOKey(InquirePendingId), mkIOKey(OutputAscii),
    mkIOKey(OutputComplex32), mkIOKey(OutputComplex64),
    mkIOKey(OutputDerivedType), mkIOKey(OutputDescriptor),
    mkIOKey(OutputInteger8), mkIOKey(OutputInteger16), mkIOKey(OutputInteger32),
    mkIOKey(OutputInteger64), mkIOKey(OutputInteger128), mkIOKey(OutputLogical),
    mkIOKey(OutputNamelist), mkIOKey(OutputReal32), mkIOKey(OutputReal64),
    mkIOKey(OutputUnformattedBlock), mkIOKey(SetAccess), mkIOKey(SetAction),
    mkIOKey(SetAdvance), mkIOKey(SetAsynchronous), mkIOKey(SetBlank),
    mkIOKey(SetCarriagecontrol), mkIOKey(SetConvert), mkIOKey(SetDecimal),
    mkIOKey(SetDelim), mkIOKey(SetEncoding), mkIOKey(SetFile), mkIOKey(SetForm),
    mkIOKey(SetPad), mkIOKey(SetPos), mkIOKey(SetPosition), mkIOKey(SetRec),
    mkIOKey(SetRecl), mkIOKey(SetRound), mkIOKey(SetSign), mkIOKey(SetStatus)>
    newIOTable;
} // namespace Fortran::lower

namespace {
/// IO statements may require exceptional condition handling. A statement that
/// encounters an exceptional condition may branch to a label given on an ERR
/// (error), END (end-of-file), or EOR (end-of-record) specifier. An IOSTAT
/// specifier variable may be set to a value that indicates some condition,
/// and an IOMSG specifier variable may be set to a description of a condition.
struct ConditionSpecInfo {
  const Fortran::lower::SomeExpr *ioStatExpr{};
  std::optional<fir::ExtendedValue> ioMsg;
  bool hasErr{};
  bool hasEnd{};
  bool hasEor{};
  fir::IfOp bigUnitIfOp;

  /// Check for any condition specifier that applies to specifier processing.
  bool hasErrorConditionSpec() const { return ioStatExpr != nullptr || hasErr; }

  /// Check for any condition specifier that applies to data transfer items
  /// in a PRINT, READ, WRITE, or WAIT statement. (WAIT may be irrelevant.)
  bool hasTransferConditionSpec() const {
    return hasErrorConditionSpec() || hasEnd || hasEor;
  }

  /// Check for any condition specifier, including IOMSG.
  bool hasAnyConditionSpec() const {
    return hasTransferConditionSpec() || ioMsg;
  }
};
} // namespace

template <typename D>
static void genIoLoop(Fortran::lower::AbstractConverter &converter,
                      mlir::Value cookie, const D &ioImpliedDo,
                      bool isFormatted, bool checkResult, mlir::Value &ok,
                      bool inLoop);

/// Helper function to retrieve the name of the IO function given the key `A`
template <typename A>
static constexpr const char *getName() {
  return std::get<A>(Fortran::lower::newIOTable).name;
}

/// Helper function to retrieve the type model signature builder of the IO
/// function as defined by the key `A`
template <typename A>
static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
  return std::get<A>(Fortran::lower::newIOTable).getTypeModel();
}

inline int64_t getLength(mlir::Type argTy) {
  return argTy.cast<fir::SequenceType>().getShape()[0];
}

/// Get (or generate) the MLIR FuncOp for a given IO runtime function.
template <typename E>
static mlir::func::FuncOp getIORuntimeFunc(mlir::Location loc,
                                           fir::FirOpBuilder &builder) {
  llvm::StringRef name = getName<E>();
  mlir::func::FuncOp func = builder.getNamedFunction(name);
  if (func)
    return func;
  auto funTy = getTypeModel<E>()(builder.getContext());
  func = builder.createFunction(loc, name, funTy);
  func->setAttr(fir::FIROpsDialect::getFirRuntimeAttrName(),
                builder.getUnitAttr());
  func->setAttr("fir.io", builder.getUnitAttr());
  return func;
}

/// Generate calls to end an IO statement. Return the IOSTAT value, if any.
/// It is the caller's responsibility to generate branches on that value.
static mlir::Value genEndIO(Fortran::lower::AbstractConverter &converter,
                            mlir::Location loc, mlir::Value cookie,
                            ConditionSpecInfo &csi,
                            Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (csi.ioMsg) {
    mlir::func::FuncOp getIoMsg =
        getIORuntimeFunc<mkIOKey(GetIoMsg)>(loc, builder);
    builder.create<fir::CallOp>(
        loc, getIoMsg,
        mlir::ValueRange{
            cookie,
            builder.createConvert(loc, getIoMsg.getFunctionType().getInput(1),
                                  fir::getBase(*csi.ioMsg)),
            builder.createConvert(loc, getIoMsg.getFunctionType().getInput(2),
                                  fir::getLen(*csi.ioMsg))});
  }
  mlir::func::FuncOp endIoStatement =
      getIORuntimeFunc<mkIOKey(EndIoStatement)>(loc, builder);
  auto call = builder.create<fir::CallOp>(loc, endIoStatement,
                                          mlir::ValueRange{cookie});
  mlir::Value iostat = call.getResult(0);
  if (csi.bigUnitIfOp) {
    stmtCtx.finalizeAndPop();
    builder.create<fir::ResultOp>(loc, iostat);
    builder.setInsertionPointAfter(csi.bigUnitIfOp);
    iostat = csi.bigUnitIfOp.getResult(0);
  }
  if (csi.ioStatExpr) {
    mlir::Value ioStatVar =
        fir::getBase(converter.genExprAddr(loc, csi.ioStatExpr, stmtCtx));
    mlir::Value ioStatResult =
        builder.createConvert(loc, converter.genType(*csi.ioStatExpr), iostat);
    builder.create<fir::StoreOp>(loc, ioStatResult, ioStatVar);
  }
  return csi.hasTransferConditionSpec() ? iostat : mlir::Value{};
}

/// Make the next call in the IO statement conditional on runtime result `ok`.
/// If a call returns `ok==false`, further suboperation calls for an IO
/// statement will be skipped. This may generate branch heavy, deeply nested
/// conditionals for IO statements with a large number of suboperations.
static void makeNextConditionalOn(fir::FirOpBuilder &builder,
                                  mlir::Location loc, bool checkResult,
                                  mlir::Value ok, bool inLoop = false) {
  if (!checkResult || !ok)
    // Either no IO calls need to be checked, or this will be the first call.
    return;

  // A previous IO call for a statement returned the bool `ok`. If this call
  // is in a fir.iterate_while loop, the result must be propagated up to the
  // loop scope as an extra ifOp result. (The propagation is done in genIoLoop.)
  mlir::TypeRange resTy;
  if (inLoop)
    resTy = builder.getI1Type();
  auto ifOp = builder.create<fir::IfOp>(loc, resTy, ok,
                                        /*withElseRegion=*/inLoop);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
}

// Derived type symbols may each be mapped to up to 4 defined IO procedures.
using DefinedIoProcMap = std::multimap<const Fortran::semantics::Symbol *,
                                       Fortran::semantics::NonTbpDefinedIo>;

/// Get the current scope's non-type-bound defined IO procedures.
static DefinedIoProcMap
getDefinedIoProcMap(Fortran::lower::AbstractConverter &converter) {
  const Fortran::semantics::Scope *scope = &converter.getCurrentScope();
  for (; !scope->IsGlobal(); scope = &scope->parent())
    if (scope->kind() == Fortran::semantics::Scope::Kind::MainProgram ||
        scope->kind() == Fortran::semantics::Scope::Kind::Subprogram ||
        scope->kind() == Fortran::semantics::Scope::Kind::BlockConstruct)
      break;
  return Fortran::semantics::CollectNonTbpDefinedIoGenericInterfaces(*scope,
                                                                     false);
}

/// Check a set of defined IO procedures for any procedure pointer or dummy
/// procedures.
static bool hasLocalDefinedIoProc(DefinedIoProcMap &definedIoProcMap) {
  for (auto &iface : definedIoProcMap) {
    const Fortran::semantics::Symbol *procSym = iface.second.subroutine;
    if (!procSym)
      continue;
    procSym = &procSym->GetUltimate();
    if (Fortran::semantics::IsProcedurePointer(*procSym) ||
        Fortran::semantics::IsDummy(*procSym))
      return true;
  }
  return false;
}

/// Retrieve or generate a runtime description of the non-type-bound defined
/// IO procedures in the current scope. If any procedure is a dummy or a
/// procedure pointer, the result is local. Otherwise the result is static.
/// If there are no procedures, return a scope-independent default table with
/// an empty procedure list, but with the `ignoreNonTbpEntries` flag set. The
/// form of the description is defined in runtime header file non-tbp-dio.h.
static mlir::Value
getNonTbpDefinedIoTableAddr(Fortran::lower::AbstractConverter &converter,
                            DefinedIoProcMap &definedIoProcMap) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::MLIRContext *context = builder.getContext();
  mlir::Location loc = converter.getCurrentLocation();
  mlir::Type refTy = fir::ReferenceType::get(mlir::NoneType::get(context));
  std::string suffix = ".nonTbpDefinedIoTable";
  std::string tableMangleName = definedIoProcMap.empty()
                                    ? "default" + suffix
                                    : converter.mangleName(suffix);
  if (auto table = builder.getNamedGlobal(tableMangleName))
    return builder.createConvert(
        loc, refTy,
        builder.create<fir::AddrOfOp>(loc, table.resultType(),
                                      table.getSymbol()));

  mlir::StringAttr linkOnce = builder.createLinkOnceLinkage();
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type sizeTy =
      fir::runtime::getModel<std::size_t>()(builder.getContext());
  mlir::Type intTy = fir::runtime::getModel<int>()(builder.getContext());
  mlir::Type boolTy = fir::runtime::getModel<bool>()(builder.getContext());
  mlir::Type listTy = fir::SequenceType::get(
      definedIoProcMap.size(),
      mlir::TupleType::get(context, {refTy, refTy, intTy, boolTy}));
  mlir::Type tableTy = mlir::TupleType::get(
      context, {sizeTy, fir::ReferenceType::get(listTy), boolTy});

  // Define the list of NonTbpDefinedIo procedures.
  bool tableIsLocal =
      !definedIoProcMap.empty() && hasLocalDefinedIoProc(definedIoProcMap);
  mlir::Value listAddr =
      tableIsLocal ? builder.create<fir::AllocaOp>(loc, listTy) : mlir::Value{};
  std::string listMangleName = tableMangleName + ".list";
  auto listFunc = [&](fir::FirOpBuilder &builder) {
    mlir::Value list = builder.create<fir::UndefOp>(loc, listTy);
    mlir::IntegerAttr intAttr[4];
    for (int i = 0; i < 4; ++i)
      intAttr[i] = builder.getIntegerAttr(idxTy, i);
    llvm::SmallVector<mlir::Attribute, 2> idx = {mlir::Attribute{},
                                                 mlir::Attribute{}};
    int n0 = 0, n1;
    auto insert = [&](mlir::Value val) {
      idx[1] = intAttr[n1++];
      list = builder.create<fir::InsertValueOp>(loc, listTy, list, val,
                                                builder.getArrayAttr(idx));
    };
    for (auto &iface : definedIoProcMap) {
      idx[0] = builder.getIntegerAttr(idxTy, n0++);
      n1 = 0;
      // derived type description [const typeInfo::DerivedType &derivedType]
      const Fortran::semantics::Symbol &dtSym = iface.first->GetUltimate();
      std::string dtName = converter.mangleName(dtSym);
      insert(builder.createConvert(
          loc, refTy,
          builder.create<fir::AddrOfOp>(
              loc, fir::ReferenceType::get(converter.genType(dtSym)),
              builder.getSymbolRefAttr(dtName))));
      // defined IO procedure [void (*subroutine)()], may be null
      const Fortran::semantics::Symbol *procSym = iface.second.subroutine;
      if (procSym) {
        procSym = &procSym->GetUltimate();
        if (Fortran::semantics::IsProcedurePointer(*procSym)) {
          TODO(loc, "defined IO procedure pointers");
        } else if (Fortran::semantics::IsDummy(*procSym)) {
          Fortran::lower::StatementContext stmtCtx;
          insert(builder.create<fir::BoxAddrOp>(
              loc, refTy,
              fir::getBase(converter.genExprAddr(
                  loc,
                  Fortran::lower::SomeExpr{
                      Fortran::evaluate::ProcedureDesignator{*procSym}},
                  stmtCtx))));
        } else {
          std::string procName = converter.mangleName(*procSym);
          mlir::func::FuncOp procDef = builder.getNamedFunction(procName);
          if (!procDef)
            procDef = Fortran::lower::getOrDeclareFunction(
                procName, Fortran::evaluate::ProcedureDesignator{*procSym},
                converter);
          insert(
              builder.createConvert(loc, refTy,
                                    builder.create<fir::AddrOfOp>(
                                        loc, procDef.getFunctionType(),
                                        builder.getSymbolRefAttr(procName))));
        }
      } else {
        insert(builder.createNullConstant(loc, refTy));
      }
      // defined IO variant, one of (read/write, formatted/unformatted)
      // [common::DefinedIo definedIo]
      insert(builder.createIntegerConstant(
          loc, intTy, static_cast<int>(iface.second.definedIo)));
      // polymorphic flag is set if first defined IO dummy arg is CLASS(T)
      // [bool isDtvArgPolymorphic]
      insert(builder.createIntegerConstant(loc, boolTy,
                                           iface.second.isDtvArgPolymorphic));
    }
    if (tableIsLocal)
      builder.create<fir::StoreOp>(loc, list, listAddr);
    else
      builder.create<fir::HasValueOp>(loc, list);
  };
  if (!definedIoProcMap.empty()) {
    if (tableIsLocal)
      listFunc(builder);
    else
      builder.createGlobalConstant(loc, listTy, listMangleName, listFunc,
                                   linkOnce);
  }

  // Define the NonTbpDefinedIoTable.
  mlir::Value tableAddr = tableIsLocal
                              ? builder.create<fir::AllocaOp>(loc, tableTy)
                              : mlir::Value{};
  auto tableFunc = [&](fir::FirOpBuilder &builder) {
    mlir::Value table = builder.create<fir::UndefOp>(loc, tableTy);
    // list item count [std::size_t items]
    table = builder.create<fir::InsertValueOp>(
        loc, tableTy, table,
        builder.createIntegerConstant(loc, sizeTy, definedIoProcMap.size()),
        builder.getArrayAttr(builder.getIntegerAttr(idxTy, 0)));
    // item list [const NonTbpDefinedIo *item]
    if (definedIoProcMap.empty())
      listAddr = builder.createNullConstant(loc, builder.getRefType(listTy));
    else if (fir::GlobalOp list = builder.getNamedGlobal(listMangleName))
      listAddr = builder.create<fir::AddrOfOp>(loc, list.resultType(),
                                               list.getSymbol());
    assert(listAddr && "missing namelist object list");
    table = builder.create<fir::InsertValueOp>(
        loc, tableTy, table, listAddr,
        builder.getArrayAttr(builder.getIntegerAttr(idxTy, 1)));
    // [bool ignoreNonTbpEntries] conservatively set to true
    table = builder.create<fir::InsertValueOp>(
        loc, tableTy, table, builder.createIntegerConstant(loc, boolTy, true),
        builder.getArrayAttr(builder.getIntegerAttr(idxTy, 2)));
    if (tableIsLocal)
      builder.create<fir::StoreOp>(loc, table, tableAddr);
    else
      builder.create<fir::HasValueOp>(loc, table);
  };
  if (tableIsLocal) {
    tableFunc(builder);
  } else {
    fir::GlobalOp table = builder.createGlobal(
        loc, tableTy, tableMangleName,
        /*isConst=*/true, /*isTarget=*/false, tableFunc, linkOnce);
    tableAddr = builder.create<fir::AddrOfOp>(
        loc, fir::ReferenceType::get(tableTy), table.getSymbol());
  }
  assert(tableAddr && "missing NonTbpDefinedIo table result");
  return builder.createConvert(loc, refTy, tableAddr);
}

static mlir::Value
getNonTbpDefinedIoTableAddr(Fortran::lower::AbstractConverter &converter) {
  DefinedIoProcMap definedIoProcMap = getDefinedIoProcMap(converter);
  return getNonTbpDefinedIoTableAddr(converter, definedIoProcMap);
}

/// Retrieve or generate a runtime description of NAMELIST group \p symbol.
/// The form of the description is defined in runtime header file namelist.h.
/// Static descriptors are generated for global objects; local descriptors for
/// local objects. If all descriptors and defined IO procedures are static,
/// the NamelistGroup is static.
static mlir::Value
getNamelistGroup(Fortran::lower::AbstractConverter &converter,
                 const Fortran::semantics::Symbol &symbol,
                 Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  std::string groupMangleName = converter.mangleName(symbol);
  if (auto group = builder.getNamedGlobal(groupMangleName))
    return builder.create<fir::AddrOfOp>(loc, group.resultType(),
                                         group.getSymbol());

  const auto &details =
      symbol.GetUltimate().get<Fortran::semantics::NamelistDetails>();
  mlir::MLIRContext *context = builder.getContext();
  mlir::StringAttr linkOnce = builder.createLinkOnceLinkage();
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type sizeTy =
      fir::runtime::getModel<std::size_t>()(builder.getContext());
  mlir::Type charRefTy = fir::ReferenceType::get(builder.getIntegerType(8));
  mlir::Type descRefTy =
      fir::ReferenceType::get(fir::BoxType::get(mlir::NoneType::get(context)));
  mlir::Type listTy = fir::SequenceType::get(
      details.objects().size(),
      mlir::TupleType::get(context, {charRefTy, descRefTy}));
  mlir::Type groupTy = mlir::TupleType::get(
      context, {charRefTy, sizeTy, fir::ReferenceType::get(listTy),
                fir::ReferenceType::get(mlir::NoneType::get(context))});
  auto stringAddress = [&](const Fortran::semantics::Symbol &symbol) {
    return fir::factory::createStringLiteral(builder, loc,
                                             symbol.name().ToString() + '\0');
  };

  // Define variable names, and static descriptors for global variables.
  DefinedIoProcMap definedIoProcMap = getDefinedIoProcMap(converter);
  bool groupIsLocal = hasLocalDefinedIoProc(definedIoProcMap);
  stringAddress(symbol);
  for (const Fortran::semantics::Symbol &s : details.objects()) {
    stringAddress(s);
    if (!Fortran::lower::symbolIsGlobal(s)) {
      groupIsLocal = true;
      continue;
    }
    // A global pointer or allocatable variable has a descriptor for typical
    // accesses. Variables in multiple namelist groups may already have one.
    // Create descriptors for other cases.
    if (!IsAllocatableOrObjectPointer(&s)) {
      std::string mangleName =
          Fortran::lower::mangle::globalNamelistDescriptorName(s);
      if (builder.getNamedGlobal(mangleName))
        continue;
      const auto expr = Fortran::evaluate::AsGenericExpr(s);
      fir::BoxType boxTy =
          fir::BoxType::get(fir::PointerType::get(converter.genType(s)));
      auto descFunc = [&](fir::FirOpBuilder &b) {
        auto box = Fortran::lower::genInitialDataTarget(
            converter, loc, boxTy, *expr, /*couldBeInEquivalence=*/true);
        b.create<fir::HasValueOp>(loc, box);
      };
      builder.createGlobalConstant(loc, boxTy, mangleName, descFunc, linkOnce);
    }
  }

  // Define the list of Items.
  mlir::Value listAddr =
      groupIsLocal ? builder.create<fir::AllocaOp>(loc, listTy) : mlir::Value{};
  std::string listMangleName = groupMangleName + ".list";
  auto listFunc = [&](fir::FirOpBuilder &builder) {
    mlir::Value list = builder.create<fir::UndefOp>(loc, listTy);
    mlir::IntegerAttr zero = builder.getIntegerAttr(idxTy, 0);
    mlir::IntegerAttr one = builder.getIntegerAttr(idxTy, 1);
    llvm::SmallVector<mlir::Attribute, 2> idx = {mlir::Attribute{},
                                                 mlir::Attribute{}};
    int n = 0;
    for (const Fortran::semantics::Symbol &s : details.objects()) {
      idx[0] = builder.getIntegerAttr(idxTy, n++);
      idx[1] = zero;
      mlir::Value nameAddr =
          builder.createConvert(loc, charRefTy, fir::getBase(stringAddress(s)));
      list = builder.create<fir::InsertValueOp>(loc, listTy, list, nameAddr,
                                                builder.getArrayAttr(idx));
      idx[1] = one;
      mlir::Value descAddr;
      if (auto desc = builder.getNamedGlobal(
              Fortran::lower::mangle::globalNamelistDescriptorName(s))) {
        descAddr = builder.create<fir::AddrOfOp>(loc, desc.resultType(),
                                                 desc.getSymbol());
      } else if (Fortran::semantics::FindCommonBlockContaining(s) &&
                 IsAllocatableOrPointer(s)) {
        mlir::Type symType = converter.genType(s);
        const Fortran::semantics::Symbol *commonBlockSym =
            Fortran::semantics::FindCommonBlockContaining(s);
        std::string commonBlockName = converter.mangleName(*commonBlockSym);
        fir::GlobalOp commonGlobal = builder.getNamedGlobal(commonBlockName);
        mlir::Value commonBlockAddr = builder.create<fir::AddrOfOp>(
            loc, commonGlobal.resultType(), commonGlobal.getSymbol());
        mlir::IntegerType i8Ty = builder.getIntegerType(8);
        mlir::Type i8Ptr = builder.getRefType(i8Ty);
        mlir::Type seqTy = builder.getRefType(builder.getVarLenSeqTy(i8Ty));
        mlir::Value base = builder.createConvert(loc, seqTy, commonBlockAddr);
        std::size_t byteOffset = s.GetUltimate().offset();
        mlir::Value offs = builder.createIntegerConstant(
            loc, builder.getIndexType(), byteOffset);
        mlir::Value varAddr = builder.create<fir::CoordinateOp>(
            loc, i8Ptr, base, mlir::ValueRange{offs});
        descAddr =
            builder.createConvert(loc, builder.getRefType(symType), varAddr);
      } else {
        const auto expr = Fortran::evaluate::AsGenericExpr(s);
        fir::ExtendedValue exv = converter.genExprAddr(*expr, stmtCtx);
        mlir::Type type = fir::getBase(exv).getType();
        if (mlir::Type baseTy = fir::dyn_cast_ptrOrBoxEleTy(type))
          type = baseTy;
        fir::BoxType boxType = fir::BoxType::get(fir::PointerType::get(type));
        descAddr = builder.createTemporary(loc, boxType);
        fir::MutableBoxValue box = fir::MutableBoxValue(descAddr, {}, {});
        fir::factory::associateMutableBox(builder, loc, box, exv,
                                          /*lbounds=*/std::nullopt);
      }
      descAddr = builder.createConvert(loc, descRefTy, descAddr);
      list = builder.create<fir::InsertValueOp>(loc, listTy, list, descAddr,
                                                builder.getArrayAttr(idx));
    }
    if (groupIsLocal)
      builder.create<fir::StoreOp>(loc, list, listAddr);
    else
      builder.create<fir::HasValueOp>(loc, list);
  };
  if (groupIsLocal)
    listFunc(builder);
  else
    builder.createGlobalConstant(loc, listTy, listMangleName, listFunc,
                                 linkOnce);

  // Define the group.
  mlir::Value groupAddr = groupIsLocal
                              ? builder.create<fir::AllocaOp>(loc, groupTy)
                              : mlir::Value{};
  auto groupFunc = [&](fir::FirOpBuilder &builder) {
    mlir::Value group = builder.create<fir::UndefOp>(loc, groupTy);
    // group name [const char *groupName]
    group = builder.create<fir::InsertValueOp>(
        loc, groupTy, group,
        builder.createConvert(loc, charRefTy,
                              fir::getBase(stringAddress(symbol))),
        builder.getArrayAttr(builder.getIntegerAttr(idxTy, 0)));
    // list item count [std::size_t items]
    group = builder.create<fir::InsertValueOp>(
        loc, groupTy, group,
        builder.createIntegerConstant(loc, sizeTy, details.objects().size()),
        builder.getArrayAttr(builder.getIntegerAttr(idxTy, 1)));
    // item list [const Item *item]
    if (fir::GlobalOp list = builder.getNamedGlobal(listMangleName))
      listAddr = builder.create<fir::AddrOfOp>(loc, list.resultType(),
                                               list.getSymbol());
    assert(listAddr && "missing namelist object list");
    group = builder.create<fir::InsertValueOp>(
        loc, groupTy, group, listAddr,
        builder.getArrayAttr(builder.getIntegerAttr(idxTy, 2)));
    // non-type-bound defined IO procedures
    // [const NonTbpDefinedIoTable *nonTbpDefinedIo]
    group = builder.create<fir::InsertValueOp>(
        loc, groupTy, group,
        getNonTbpDefinedIoTableAddr(converter, definedIoProcMap),
        builder.getArrayAttr(builder.getIntegerAttr(idxTy, 3)));
    if (groupIsLocal)
      builder.create<fir::StoreOp>(loc, group, groupAddr);
    else
      builder.create<fir::HasValueOp>(loc, group);
  };
  if (groupIsLocal) {
    groupFunc(builder);
  } else {
    fir::GlobalOp group = builder.createGlobal(
        loc, groupTy, groupMangleName,
        /*isConst=*/true, /*isTarget=*/false, groupFunc, linkOnce);
    groupAddr = builder.create<fir::AddrOfOp>(loc, group.resultType(),
                                              group.getSymbol());
  }
  assert(groupAddr && "missing namelist group result");
  return groupAddr;
}

/// Generate a namelist IO call.
static void genNamelistIO(Fortran::lower::AbstractConverter &converter,
                          mlir::Value cookie, mlir::func::FuncOp funcOp,
                          Fortran::semantics::Symbol &symbol, bool checkResult,
                          mlir::Value &ok,
                          Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  makeNextConditionalOn(builder, loc, checkResult, ok);
  mlir::Type argType = funcOp.getFunctionType().getInput(1);
  mlir::Value groupAddr = getNamelistGroup(converter, symbol, stmtCtx);
  groupAddr = builder.createConvert(loc, argType, groupAddr);
  llvm::SmallVector<mlir::Value> args = {cookie, groupAddr};
  ok = builder.create<fir::CallOp>(loc, funcOp, args).getResult(0);
}

/// Get the output function to call for a value of the given type.
static mlir::func::FuncOp getOutputFunc(mlir::Location loc,
                                        fir::FirOpBuilder &builder,
                                        mlir::Type type, bool isFormatted) {
  if (fir::unwrapPassByRefType(type).isa<fir::RecordType>())
    return getIORuntimeFunc<mkIOKey(OutputDerivedType)>(loc, builder);
  if (!isFormatted)
    return getIORuntimeFunc<mkIOKey(OutputDescriptor)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::IntegerType>()) {
    switch (ty.getWidth()) {
    case 1:
      return getIORuntimeFunc<mkIOKey(OutputLogical)>(loc, builder);
    case 8:
      return getIORuntimeFunc<mkIOKey(OutputInteger8)>(loc, builder);
    case 16:
      return getIORuntimeFunc<mkIOKey(OutputInteger16)>(loc, builder);
    case 32:
      return getIORuntimeFunc<mkIOKey(OutputInteger32)>(loc, builder);
    case 64:
      return getIORuntimeFunc<mkIOKey(OutputInteger64)>(loc, builder);
    case 128:
      return getIORuntimeFunc<mkIOKey(OutputInteger128)>(loc, builder);
    }
    llvm_unreachable("unknown OutputInteger kind");
  }
  if (auto ty = type.dyn_cast<mlir::FloatType>()) {
    if (auto width = ty.getWidth(); width == 32)
      return getIORuntimeFunc<mkIOKey(OutputReal32)>(loc, builder);
    else if (width == 64)
      return getIORuntimeFunc<mkIOKey(OutputReal64)>(loc, builder);
  }
  auto kindMap = fir::getKindMapping(builder.getModule());
  if (auto ty = type.dyn_cast<fir::ComplexType>()) {
    // COMPLEX(KIND=k) corresponds to a pair of REAL(KIND=k).
    auto width = kindMap.getRealBitsize(ty.getFKind());
    if (width == 32)
      return getIORuntimeFunc<mkIOKey(OutputComplex32)>(loc, builder);
    else if (width == 64)
      return getIORuntimeFunc<mkIOKey(OutputComplex64)>(loc, builder);
  }
  if (type.isa<fir::LogicalType>())
    return getIORuntimeFunc<mkIOKey(OutputLogical)>(loc, builder);
  if (fir::factory::CharacterExprHelper::isCharacterScalar(type)) {
    // TODO: What would it mean if the default CHARACTER KIND is set to a wide
    // character encoding scheme? How do we handle UTF-8? Is it a distinct KIND
    // value? For now, assume that if the default CHARACTER KIND is 8 bit,
    // then it is an ASCII string and UTF-8 is unsupported.
    auto asciiKind = kindMap.defaultCharacterKind();
    if (kindMap.getCharacterBitsize(asciiKind) == 8 &&
        fir::factory::CharacterExprHelper::getCharacterKind(type) == asciiKind)
      return getIORuntimeFunc<mkIOKey(OutputAscii)>(loc, builder);
  }
  return getIORuntimeFunc<mkIOKey(OutputDescriptor)>(loc, builder);
}

/// Generate a sequence of output data transfer calls.
static void genOutputItemList(
    Fortran::lower::AbstractConverter &converter, mlir::Value cookie,
    const std::list<Fortran::parser::OutputItem> &items, bool isFormatted,
    bool checkResult, mlir::Value &ok, bool inLoop) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const Fortran::parser::OutputItem &item : items) {
    if (const auto &impliedDo = std::get_if<1>(&item.u)) {
      genIoLoop(converter, cookie, impliedDo->value(), isFormatted, checkResult,
                ok, inLoop);
      continue;
    }
    auto &pExpr = std::get<Fortran::parser::Expr>(item.u);
    mlir::Location loc = converter.genLocation(pExpr.source);
    makeNextConditionalOn(builder, loc, checkResult, ok, inLoop);
    Fortran::lower::StatementContext stmtCtx;

    const auto *expr = Fortran::semantics::GetExpr(pExpr);
    if (!expr)
      fir::emitFatalError(loc, "internal error: could not get evaluate::Expr");
    mlir::Type itemTy = converter.genType(*expr);
    mlir::func::FuncOp outputFunc =
        getOutputFunc(loc, builder, itemTy, isFormatted);
    mlir::Type argType = outputFunc.getFunctionType().getInput(1);
    assert((isFormatted || argType.isa<fir::BoxType>()) &&
           "expect descriptor for unformatted IO runtime");
    llvm::SmallVector<mlir::Value> outputFuncArgs = {cookie};
    fir::factory::CharacterExprHelper helper{builder, loc};
    if (argType.isa<fir::BoxType>()) {
      mlir::Value box = fir::getBase(converter.genExprBox(loc, *expr, stmtCtx));
      outputFuncArgs.push_back(builder.createConvert(loc, argType, box));
      if (fir::unwrapPassByRefType(itemTy).isa<fir::RecordType>())
        outputFuncArgs.push_back(getNonTbpDefinedIoTableAddr(converter));
    } else if (helper.isCharacterScalar(itemTy)) {
      fir::ExtendedValue exv = converter.genExprAddr(loc, expr, stmtCtx);
      // scalar allocatable/pointer may also get here, not clear if
      // genExprAddr will lower them as CharBoxValue or BoxValue.
      if (!exv.getCharBox())
        llvm::report_fatal_error(
            "internal error: scalar character not in CharBox");
      outputFuncArgs.push_back(builder.createConvert(
          loc, outputFunc.getFunctionType().getInput(1), fir::getBase(exv)));
      outputFuncArgs.push_back(builder.createConvert(
          loc, outputFunc.getFunctionType().getInput(2), fir::getLen(exv)));
    } else {
      fir::ExtendedValue itemBox = converter.genExprValue(loc, expr, stmtCtx);
      mlir::Value itemValue = fir::getBase(itemBox);
      if (fir::isa_complex(itemTy)) {
        auto parts =
            fir::factory::Complex{builder, loc}.extractParts(itemValue);
        outputFuncArgs.push_back(parts.first);
        outputFuncArgs.push_back(parts.second);
      } else {
        itemValue = builder.createConvert(loc, argType, itemValue);
        outputFuncArgs.push_back(itemValue);
      }
    }
    ok = builder.create<fir::CallOp>(loc, outputFunc, outputFuncArgs)
             .getResult(0);
  }
}

/// Get the input function to call for a value of the given type.
static mlir::func::FuncOp getInputFunc(mlir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       mlir::Type type, bool isFormatted) {
  if (fir::unwrapPassByRefType(type).isa<fir::RecordType>())
    return getIORuntimeFunc<mkIOKey(InputDerivedType)>(loc, builder);
  if (!isFormatted)
    return getIORuntimeFunc<mkIOKey(InputDescriptor)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::IntegerType>())
    return ty.getWidth() == 1
               ? getIORuntimeFunc<mkIOKey(InputLogical)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(InputInteger)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::FloatType>()) {
    if (auto width = ty.getWidth(); width == 32)
      return getIORuntimeFunc<mkIOKey(InputReal32)>(loc, builder);
    else if (width == 64)
      return getIORuntimeFunc<mkIOKey(InputReal64)>(loc, builder);
  }
  auto kindMap = fir::getKindMapping(builder.getModule());
  if (auto ty = type.dyn_cast<fir::ComplexType>()) {
    auto width = kindMap.getRealBitsize(ty.getFKind());
    if (width == 32)
      return getIORuntimeFunc<mkIOKey(InputComplex32)>(loc, builder);
    else if (width == 64)
      return getIORuntimeFunc<mkIOKey(InputComplex64)>(loc, builder);
  }
  if (type.isa<fir::LogicalType>())
    return getIORuntimeFunc<mkIOKey(InputLogical)>(loc, builder);
  if (fir::factory::CharacterExprHelper::isCharacterScalar(type)) {
    auto asciiKind = kindMap.defaultCharacterKind();
    if (kindMap.getCharacterBitsize(asciiKind) == 8 &&
        fir::factory::CharacterExprHelper::getCharacterKind(type) == asciiKind)
      return getIORuntimeFunc<mkIOKey(InputAscii)>(loc, builder);
  }
  return getIORuntimeFunc<mkIOKey(InputDescriptor)>(loc, builder);
}

/// Interpret the lowest byte of a LOGICAL and store that value into the full
/// storage of the LOGICAL. The load, convert, and store effectively (sign or
/// zero) extends the lowest byte into the full LOGICAL value storage, as the
/// runtime is unaware of the LOGICAL value's actual bit width (it was passed
/// as a `bool&` to the runtime in order to be set).
static void boolRefToLogical(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Value addr) {
  auto boolType = builder.getRefType(builder.getI1Type());
  auto boolAddr = builder.createConvert(loc, boolType, addr);
  auto boolValue = builder.create<fir::LoadOp>(loc, boolAddr);
  auto logicalType = fir::unwrapPassByRefType(addr.getType());
  // The convert avoid making any assumptions about how LOGICALs are actually
  // represented (it might end-up being either a signed or zero extension).
  auto logicalValue = builder.createConvert(loc, logicalType, boolValue);
  builder.create<fir::StoreOp>(loc, logicalValue, addr);
}

static mlir::Value
createIoRuntimeCallForItem(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, mlir::func::FuncOp inputFunc,
                           mlir::Value cookie, const fir::ExtendedValue &item) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type argType = inputFunc.getFunctionType().getInput(1);
  llvm::SmallVector<mlir::Value> inputFuncArgs = {cookie};
  if (argType.isa<fir::BaseBoxType>()) {
    mlir::Value box = fir::getBase(item);
    auto boxTy = box.getType().dyn_cast<fir::BaseBoxType>();
    assert(boxTy && "must be previously emboxed");
    inputFuncArgs.push_back(builder.createConvert(loc, argType, box));
    if (fir::unwrapPassByRefType(boxTy).isa<fir::RecordType>())
      inputFuncArgs.push_back(getNonTbpDefinedIoTableAddr(converter));
  } else {
    mlir::Value itemAddr = fir::getBase(item);
    mlir::Type itemTy = fir::unwrapPassByRefType(itemAddr.getType());
    inputFuncArgs.push_back(builder.createConvert(loc, argType, itemAddr));
    fir::factory::CharacterExprHelper charHelper{builder, loc};
    if (charHelper.isCharacterScalar(itemTy)) {
      mlir::Value len = fir::getLen(item);
      inputFuncArgs.push_back(builder.createConvert(
          loc, inputFunc.getFunctionType().getInput(2), len));
    } else if (itemTy.isa<mlir::IntegerType>()) {
      inputFuncArgs.push_back(builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI32IntegerAttr(
                   itemTy.cast<mlir::IntegerType>().getWidth() / 8)));
    }
  }
  auto call = builder.create<fir::CallOp>(loc, inputFunc, inputFuncArgs);
  auto itemAddr = fir::getBase(item);
  auto itemTy = fir::unwrapRefType(itemAddr.getType());
  if (itemTy.isa<fir::LogicalType>())
    boolRefToLogical(loc, builder, itemAddr);
  return call.getResult(0);
}

/// Generate a sequence of input data transfer calls.
static void genInputItemList(Fortran::lower::AbstractConverter &converter,
                             mlir::Value cookie,
                             const std::list<Fortran::parser::InputItem> &items,
                             bool isFormatted, bool checkResult,
                             mlir::Value &ok, bool inLoop) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const Fortran::parser::InputItem &item : items) {
    if (const auto &impliedDo = std::get_if<1>(&item.u)) {
      genIoLoop(converter, cookie, impliedDo->value(), isFormatted, checkResult,
                ok, inLoop);
      continue;
    }
    auto &pVar = std::get<Fortran::parser::Variable>(item.u);
    mlir::Location loc = converter.genLocation(pVar.GetSource());
    makeNextConditionalOn(builder, loc, checkResult, ok, inLoop);
    Fortran::lower::StatementContext stmtCtx;
    const auto *expr = Fortran::semantics::GetExpr(pVar);
    if (!expr)
      fir::emitFatalError(loc, "internal error: could not get evaluate::Expr");
    if (Fortran::evaluate::HasVectorSubscript(*expr)) {
      auto vectorSubscriptBox =
          Fortran::lower::genVectorSubscriptBox(loc, converter, stmtCtx, *expr);
      mlir::func::FuncOp inputFunc = getInputFunc(
          loc, builder, vectorSubscriptBox.getElementType(), isFormatted);
      const bool mustBox =
          inputFunc.getFunctionType().getInput(1).isa<fir::BoxType>();
      if (!checkResult) {
        auto elementalGenerator = [&](const fir::ExtendedValue &element) {
          createIoRuntimeCallForItem(converter, loc, inputFunc, cookie,
                                     mustBox ? builder.createBox(loc, element)
                                             : element);
        };
        vectorSubscriptBox.loopOverElements(builder, loc, elementalGenerator);
      } else {
        auto elementalGenerator =
            [&](const fir::ExtendedValue &element) -> mlir::Value {
          return createIoRuntimeCallForItem(
              converter, loc, inputFunc, cookie,
              mustBox ? builder.createBox(loc, element) : element);
        };
        if (!ok)
          ok = builder.createBool(loc, true);
        ok = vectorSubscriptBox.loopOverElementsWhile(builder, loc,
                                                      elementalGenerator, ok);
      }
      continue;
    }
    mlir::Type itemTy = converter.genType(*expr);
    mlir::func::FuncOp inputFunc =
        getInputFunc(loc, builder, itemTy, isFormatted);
    auto itemExv = inputFunc.getFunctionType().getInput(1).isa<fir::BoxType>()
                       ? converter.genExprBox(loc, *expr, stmtCtx)
                       : converter.genExprAddr(loc, expr, stmtCtx);
    ok = createIoRuntimeCallForItem(converter, loc, inputFunc, cookie, itemExv);
  }
}

/// Generate an io-implied-do loop.
template <typename D>
static void genIoLoop(Fortran::lower::AbstractConverter &converter,
                      mlir::Value cookie, const D &ioImpliedDo,
                      bool isFormatted, bool checkResult, mlir::Value &ok,
                      bool inLoop) {
  Fortran::lower::StatementContext stmtCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  makeNextConditionalOn(builder, loc, checkResult, ok, inLoop);
  const auto &itemList = std::get<0>(ioImpliedDo.t);
  const auto &control = std::get<1>(ioImpliedDo.t);
  const auto &loopSym = *control.name.thing.thing.symbol;
  mlir::Value loopVar = fir::getBase(converter.genExprAddr(
      Fortran::evaluate::AsGenericExpr(loopSym).value(), stmtCtx));
  auto genControlValue = [&](const Fortran::parser::ScalarIntExpr &expr) {
    mlir::Value v = fir::getBase(
        converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
    return builder.createConvert(loc, builder.getIndexType(), v);
  };
  mlir::Value lowerValue = genControlValue(control.lower);
  mlir::Value upperValue = genControlValue(control.upper);
  mlir::Value stepValue =
      control.step.has_value()
          ? genControlValue(*control.step)
          : builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto genItemList = [&](const D &ioImpliedDo) {
    if constexpr (std::is_same_v<D, Fortran::parser::InputImpliedDo>)
      genInputItemList(converter, cookie, itemList, isFormatted, checkResult,
                       ok, /*inLoop=*/true);
    else
      genOutputItemList(converter, cookie, itemList, isFormatted, checkResult,
                        ok, /*inLoop=*/true);
  };
  if (!checkResult) {
    // No IO call result checks - the loop is a fir.do_loop op.
    auto doLoopOp = builder.create<fir::DoLoopOp>(
        loc, lowerValue, upperValue, stepValue, /*unordered=*/false,
        /*finalCountValue=*/true);
    builder.setInsertionPointToStart(doLoopOp.getBody());
    mlir::Value lcv = builder.createConvert(
        loc, fir::unwrapRefType(loopVar.getType()), doLoopOp.getInductionVar());
    builder.create<fir::StoreOp>(loc, lcv, loopVar);
    genItemList(ioImpliedDo);
    builder.setInsertionPointToEnd(doLoopOp.getBody());
    mlir::Value result = builder.create<mlir::arith::AddIOp>(
        loc, doLoopOp.getInductionVar(), doLoopOp.getStep());
    builder.create<fir::ResultOp>(loc, result);
    builder.setInsertionPointAfter(doLoopOp);
    // The loop control variable may be used after the loop.
    lcv = builder.createConvert(loc, fir::unwrapRefType(loopVar.getType()),
                                doLoopOp.getResult(0));
    builder.create<fir::StoreOp>(loc, lcv, loopVar);
    return;
  }
  // Check IO call results - the loop is a fir.iterate_while op.
  if (!ok)
    ok = builder.createBool(loc, true);
  auto iterWhileOp = builder.create<fir::IterWhileOp>(
      loc, lowerValue, upperValue, stepValue, ok, /*finalCountValue*/ true);
  builder.setInsertionPointToStart(iterWhileOp.getBody());
  mlir::Value lcv =
      builder.createConvert(loc, fir::unwrapRefType(loopVar.getType()),
                            iterWhileOp.getInductionVar());
  builder.create<fir::StoreOp>(loc, lcv, loopVar);
  ok = iterWhileOp.getIterateVar();
  mlir::Value falseValue =
      builder.createIntegerConstant(loc, builder.getI1Type(), 0);
  genItemList(ioImpliedDo);
  // Unwind nested IO call scopes, filling in true and false ResultOp's.
  for (mlir::Operation *op = builder.getBlock()->getParentOp();
       mlir::isa<fir::IfOp>(op); op = op->getBlock()->getParentOp()) {
    auto ifOp = mlir::dyn_cast<fir::IfOp>(op);
    mlir::Operation *lastOp = &ifOp.getThenRegion().front().back();
    builder.setInsertionPointAfter(lastOp);
    // The primary ifOp result is the result of an IO call or loop.
    if (mlir::isa<fir::CallOp, fir::IfOp>(*lastOp))
      builder.create<fir::ResultOp>(loc, lastOp->getResult(0));
    else
      builder.create<fir::ResultOp>(loc, ok); // loop result
    // The else branch propagates an early exit false result.
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    builder.create<fir::ResultOp>(loc, falseValue);
  }
  builder.setInsertionPointToEnd(iterWhileOp.getBody());
  mlir::OpResult iterateResult = builder.getBlock()->back().getResult(0);
  mlir::Value inductionResult0 = iterWhileOp.getInductionVar();
  auto inductionResult1 = builder.create<mlir::arith::AddIOp>(
      loc, inductionResult0, iterWhileOp.getStep());
  auto inductionResult = builder.create<mlir::arith::SelectOp>(
      loc, iterateResult, inductionResult1, inductionResult0);
  llvm::SmallVector<mlir::Value> results = {inductionResult, iterateResult};
  builder.create<fir::ResultOp>(loc, results);
  ok = iterWhileOp.getResult(1);
  builder.setInsertionPointAfter(iterWhileOp);
  // The loop control variable may be used after the loop.
  lcv = builder.createConvert(loc, fir::unwrapRefType(loopVar.getType()),
                              iterWhileOp.getResult(0));
  builder.create<fir::StoreOp>(loc, lcv, loopVar);
}

//===----------------------------------------------------------------------===//
// Default argument generation.
//===----------------------------------------------------------------------===//

static mlir::Value locToFilename(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc, mlir::Type toType) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  return builder.createConvert(loc, toType,
                               fir::factory::locationToFilename(builder, loc));
}

static mlir::Value locToLineNo(Fortran::lower::AbstractConverter &converter,
                               mlir::Location loc, mlir::Type toType) {
  return fir::factory::locationToLineNo(converter.getFirOpBuilder(), loc,
                                        toType);
}

static mlir::Value getDefaultScratch(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Type toType) {
  mlir::Value null = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(0));
  return builder.createConvert(loc, toType, null);
}

static mlir::Value getDefaultScratchLen(fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Type toType) {
  return builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIntegerAttr(toType, 0));
}

/// Generate a reference to a buffer and the length of buffer given
/// a character expression. An array expression will be cast to scalar
/// character as long as they are contiguous.
static std::tuple<mlir::Value, mlir::Value>
genBuffer(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const Fortran::lower::SomeExpr &expr, mlir::Type strTy,
          mlir::Type lenTy, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  fir::ExtendedValue exprAddr = converter.genExprAddr(expr, stmtCtx);
  fir::factory::CharacterExprHelper helper(builder, loc);
  using ValuePair = std::pair<mlir::Value, mlir::Value>;
  auto [buff, len] = exprAddr.match(
      [&](const fir::CharBoxValue &x) -> ValuePair {
        return {x.getBuffer(), x.getLen()};
      },
      [&](const fir::CharArrayBoxValue &x) -> ValuePair {
        fir::CharBoxValue scalar = helper.toScalarCharacter(x);
        return {scalar.getBuffer(), scalar.getLen()};
      },
      [&](const fir::BoxValue &) -> ValuePair {
        // May need to copy before after IO to handle contiguous
        // aspect. Not sure descriptor can get here though.
        TODO(loc, "character descriptor to contiguous buffer");
      },
      [&](const auto &) -> ValuePair {
        llvm::report_fatal_error(
            "internal error: IO buffer is not a character");
      });
  buff = builder.createConvert(loc, strTy, buff);
  len = builder.createConvert(loc, lenTy, len);
  return {buff, len};
}

/// Lower a string literal. Many arguments to the runtime are conveyed as
/// Fortran CHARACTER literals.
template <typename A>
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerStringLit(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
               Fortran::lower::StatementContext &stmtCtx, const A &syntax,
               mlir::Type strTy, mlir::Type lenTy, mlir::Type ty2 = {}) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto *expr = Fortran::semantics::GetExpr(syntax);
  if (!expr)
    fir::emitFatalError(loc, "internal error: null semantic expr in IO");
  auto [buff, len] = genBuffer(converter, loc, *expr, strTy, lenTy, stmtCtx);
  mlir::Value kind;
  if (ty2) {
    auto kindVal = expr->GetType().value().kind();
    kind = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(ty2, kindVal));
  }
  return {buff, len, kind};
}

/// Pass the body of the FORMAT statement in as if it were a CHARACTER literal
/// constant. NB: This is the prescribed manner in which the front-end passes
/// this information to lowering.
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerSourceTextAsStringLit(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, llvm::StringRef text,
                           mlir::Type strTy, mlir::Type lenTy) {
  text = text.drop_front(text.find('('));
  text = text.take_front(text.rfind(')') + 1);
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Value addrGlobalStringLit =
      fir::getBase(fir::factory::createStringLiteral(builder, loc, text));
  mlir::Value buff = builder.createConvert(loc, strTy, addrGlobalStringLit);
  mlir::Value len = builder.createIntegerConstant(loc, lenTy, text.size());
  return {buff, len, mlir::Value{}};
}

//===----------------------------------------------------------------------===//
// Handle IO statement specifiers.
// These are threaded together for a single statement via the passed cookie.
//===----------------------------------------------------------------------===//

/// Generic to build an integral argument to the runtime.
template <typename A, typename B>
mlir::Value genIntIOOption(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, mlir::Value cookie,
                           const B &spec) {
  Fortran::lower::StatementContext localStatementCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::func::FuncOp ioFunc = getIORuntimeFunc<A>(loc, builder);
  mlir::FunctionType ioFuncTy = ioFunc.getFunctionType();
  mlir::Value expr = fir::getBase(converter.genExprValue(
      loc, Fortran::semantics::GetExpr(spec.v), localStatementCtx));
  mlir::Value val = builder.createConvert(loc, ioFuncTy.getInput(1), expr);
  llvm::SmallVector<mlir::Value> ioArgs = {cookie, val};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

/// Generic to build a string argument to the runtime. This passes a CHARACTER
/// as a pointer to the buffer and a LEN parameter.
template <typename A, typename B>
mlir::Value genCharIOOption(Fortran::lower::AbstractConverter &converter,
                            mlir::Location loc, mlir::Value cookie,
                            const B &spec) {
  Fortran::lower::StatementContext localStatementCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::func::FuncOp ioFunc = getIORuntimeFunc<A>(loc, builder);
  mlir::FunctionType ioFuncTy = ioFunc.getFunctionType();
  std::tuple<mlir::Value, mlir::Value, mlir::Value> tup =
      lowerStringLit(converter, loc, localStatementCtx, spec,
                     ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value> ioArgs = {cookie, std::get<0>(tup),
                                           std::get<1>(tup)};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <typename A>
mlir::Value genIOOption(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie, const A &spec) {
  // These specifiers are processed in advance elsewhere - skip them here.
  using PreprocessedSpecs =
      std::tuple<Fortran::parser::EndLabel, Fortran::parser::EorLabel,
                 Fortran::parser::ErrLabel, Fortran::parser::FileUnitNumber,
                 Fortran::parser::Format, Fortran::parser::IoUnit,
                 Fortran::parser::MsgVariable, Fortran::parser::Name,
                 Fortran::parser::StatVariable>;
  static_assert(Fortran::common::HasMember<A, PreprocessedSpecs>,
                "missing genIOOPtion specialization");
  return {};
}

template <>
mlir::Value genIOOption<Fortran::parser::FileNameExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::FileNameExpr &spec) {
  Fortran::lower::StatementContext localStatementCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  // has an extra KIND argument
  mlir::func::FuncOp ioFunc = getIORuntimeFunc<mkIOKey(SetFile)>(loc, builder);
  mlir::FunctionType ioFuncTy = ioFunc.getFunctionType();
  std::tuple<mlir::Value, mlir::Value, mlir::Value> tup =
      lowerStringLit(converter, loc, localStatementCtx, spec,
                     ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value> ioArgs{cookie, std::get<0>(tup),
                                        std::get<1>(tup)};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::ConnectSpec::CharExpr &spec) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::func::FuncOp ioFunc;
  switch (std::get<Fortran::parser::ConnectSpec::CharExpr::Kind>(spec.t)) {
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Access:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAccess)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Action:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAction)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Asynchronous:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAsynchronous)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Blank:
    ioFunc = getIORuntimeFunc<mkIOKey(SetBlank)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Decimal:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDecimal)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Delim:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDelim)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Encoding:
    ioFunc = getIORuntimeFunc<mkIOKey(SetEncoding)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Form:
    ioFunc = getIORuntimeFunc<mkIOKey(SetForm)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Pad:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPad)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Position:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPosition)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Round:
    ioFunc = getIORuntimeFunc<mkIOKey(SetRound)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Sign:
    ioFunc = getIORuntimeFunc<mkIOKey(SetSign)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Carriagecontrol:
    ioFunc = getIORuntimeFunc<mkIOKey(SetCarriagecontrol)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Convert:
    ioFunc = getIORuntimeFunc<mkIOKey(SetConvert)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Dispose:
    TODO(loc, "DISPOSE not part of the runtime::io interface");
  }
  Fortran::lower::StatementContext localStatementCtx;
  mlir::FunctionType ioFuncTy = ioFunc.getFunctionType();
  std::tuple<mlir::Value, mlir::Value, mlir::Value> tup =
      lowerStringLit(converter, loc, localStatementCtx,
                     std::get<Fortran::parser::ScalarDefaultCharExpr>(spec.t),
                     ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value> ioArgs = {cookie, std::get<0>(tup),
                                           std::get<1>(tup)};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::Recl>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::ConnectSpec::Recl &spec) {
  return genIntIOOption<mkIOKey(SetRecl)>(converter, loc, cookie, spec);
}

template <>
mlir::Value genIOOption<Fortran::parser::StatusExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::StatusExpr &spec) {
  return genCharIOOption<mkIOKey(SetStatus)>(converter, loc, cookie, spec.v);
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::CharExpr &spec) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::func::FuncOp ioFunc;
  switch (std::get<Fortran::parser::IoControlSpec::CharExpr::Kind>(spec.t)) {
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Advance:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAdvance)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Blank:
    ioFunc = getIORuntimeFunc<mkIOKey(SetBlank)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Decimal:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDecimal)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Delim:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDelim)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Pad:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPad)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Round:
    ioFunc = getIORuntimeFunc<mkIOKey(SetRound)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Sign:
    ioFunc = getIORuntimeFunc<mkIOKey(SetSign)>(loc, builder);
    break;
  }
  Fortran::lower::StatementContext localStatementCtx;
  mlir::FunctionType ioFuncTy = ioFunc.getFunctionType();
  std::tuple<mlir::Value, mlir::Value, mlir::Value> tup =
      lowerStringLit(converter, loc, localStatementCtx,
                     std::get<Fortran::parser::ScalarDefaultCharExpr>(spec.t),
                     ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value> ioArgs = {cookie, std::get<0>(tup),
                                           std::get<1>(tup)};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Asynchronous>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie,
    const Fortran::parser::IoControlSpec::Asynchronous &spec) {
  return genCharIOOption<mkIOKey(SetAsynchronous)>(converter, loc, cookie,
                                                   spec.v);
}

template <>
mlir::Value genIOOption<Fortran::parser::IdVariable>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IdVariable &spec) {
  TODO(loc, "asynchronous ID not implemented");
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Pos>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::Pos &spec) {
  return genIntIOOption<mkIOKey(SetPos)>(converter, loc, cookie, spec);
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Rec>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::Rec &spec) {
  return genIntIOOption<mkIOKey(SetRec)>(converter, loc, cookie, spec);
}

/// Generate runtime call to query the read size after an input statement if
/// the statement has SIZE control-spec.
template <typename A>
static void genIOReadSize(Fortran::lower::AbstractConverter &converter,
                          mlir::Location loc, mlir::Value cookie,
                          const A &specList, bool checkResult) {
  // This call is not conditional on the current IO status (ok) because the size
  // needs to be filled even if some error condition (end-of-file...) was met
  // during the input statement (in which case the runtime may return zero for
  // the size read).
  for (const auto &spec : specList)
    if (const auto *size =
            std::get_if<Fortran::parser::IoControlSpec::Size>(&spec.u)) {

      fir::FirOpBuilder &builder = converter.getFirOpBuilder();
      mlir::func::FuncOp ioFunc =
          getIORuntimeFunc<mkIOKey(GetSize)>(loc, builder);
      auto sizeValue =
          builder.create<fir::CallOp>(loc, ioFunc, mlir::ValueRange{cookie})
              .getResult(0);
      Fortran::lower::StatementContext localStatementCtx;
      fir::ExtendedValue var = converter.genExprAddr(
          loc, Fortran::semantics::GetExpr(size->v), localStatementCtx);
      mlir::Value varAddr = fir::getBase(var);
      mlir::Type varType = fir::unwrapPassByRefType(varAddr.getType());
      mlir::Value sizeCast = builder.createConvert(loc, varType, sizeValue);
      builder.create<fir::StoreOp>(loc, sizeCast, varAddr);
      break;
    }
}

//===----------------------------------------------------------------------===//
// Gather IO statement condition specifier information (if any).
//===----------------------------------------------------------------------===//

template <typename SEEK, typename A>
static bool hasX(const A &list) {
  for (const auto &spec : list)
    if (std::holds_alternative<SEEK>(spec.u))
      return true;
  return false;
}

template <typename SEEK, typename A>
static bool hasSpec(const A &stmt) {
  return hasX<SEEK>(stmt.v);
}

/// Get the sought expression from the specifier list.
template <typename SEEK, typename A>
static const Fortran::lower::SomeExpr *getExpr(const A &stmt) {
  for (const auto &spec : stmt.v)
    if (auto *f = std::get_if<SEEK>(&spec.u))
      return Fortran::semantics::GetExpr(f->v);
  llvm::report_fatal_error("must have a file unit");
}

/// For each specifier, build the appropriate call, threading the cookie.
template <typename A>
static void threadSpecs(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie,
                        const A &specList, bool checkResult, mlir::Value &ok) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const auto &spec : specList) {
    makeNextConditionalOn(builder, loc, checkResult, ok);
    ok = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::IoControlSpec::Size &x) -> mlir::Value {
              // Size must be queried after the related READ runtime calls, not
              // before.
              return ok;
            },
            [&](const Fortran::parser::ConnectSpec::Newunit &x) -> mlir::Value {
              // Newunit must be queried after OPEN specifier runtime calls
              // that may fail to avoid modifying the newunit variable if
              // there is an error.
              return ok;
            },
            [&](const auto &x) {
              return genIOOption(converter, loc, cookie, x);
            }},
        spec.u);
  }
}

/// Most IO statements allow one or more of five optional exception condition
/// handling specifiers: ERR, EOR, END, IOSTAT, and IOMSG. The first three
/// cause control flow to transfer to another statement. The final two return
/// information from the runtime, via a variable, about the nature of the
/// condition that occurred. These condition specifiers are handled here.
template <typename A>
ConditionSpecInfo lowerErrorSpec(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc, const A &specList) {
  ConditionSpecInfo csi;
  const Fortran::lower::SomeExpr *ioMsgExpr = nullptr;
  for (const auto &spec : specList) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::StatVariable &var) {
              csi.ioStatExpr = Fortran::semantics::GetExpr(var);
            },
            [&](const Fortran::parser::InquireSpec::IntVar &var) {
              if (std::get<Fortran::parser::InquireSpec::IntVar::Kind>(var.t) ==
                  Fortran::parser::InquireSpec::IntVar::Kind::Iostat)
                csi.ioStatExpr = Fortran::semantics::GetExpr(
                    std::get<Fortran::parser::ScalarIntVariable>(var.t));
            },
            [&](const Fortran::parser::MsgVariable &var) {
              ioMsgExpr = Fortran::semantics::GetExpr(var);
            },
            [&](const Fortran::parser::InquireSpec::CharVar &var) {
              if (std::get<Fortran::parser::InquireSpec::CharVar::Kind>(
                      var.t) ==
                  Fortran::parser::InquireSpec::CharVar::Kind::Iomsg)
                ioMsgExpr = Fortran::semantics::GetExpr(
                    std::get<Fortran::parser::ScalarDefaultCharVariable>(
                        var.t));
            },
            [&](const Fortran::parser::EndLabel &) { csi.hasEnd = true; },
            [&](const Fortran::parser::EorLabel &) { csi.hasEor = true; },
            [&](const Fortran::parser::ErrLabel &) { csi.hasErr = true; },
            [](const auto &) {}},
        spec.u);
  }
  if (ioMsgExpr) {
    // iomsg is a variable, its evaluation may require temps, but it cannot
    // itself be a temp, and it is ok to us a local statement context here.
    Fortran::lower::StatementContext stmtCtx;
    csi.ioMsg = converter.genExprAddr(loc, ioMsgExpr, stmtCtx);
  }

  return csi;
}
template <typename A>
static void
genConditionHandlerCall(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie,
                        const A &specList, ConditionSpecInfo &csi) {
  if (!csi.hasAnyConditionSpec())
    return;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::func::FuncOp enableHandlers =
      getIORuntimeFunc<mkIOKey(EnableHandlers)>(loc, builder);
  mlir::Type boolType = enableHandlers.getFunctionType().getInput(1);
  auto boolValue = [&](bool specifierIsPresent) {
    return builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(boolType, specifierIsPresent));
  };
  llvm::SmallVector<mlir::Value> ioArgs = {cookie,
                                           boolValue(csi.ioStatExpr != nullptr),
                                           boolValue(csi.hasErr),
                                           boolValue(csi.hasEnd),
                                           boolValue(csi.hasEor),
                                           boolValue(csi.ioMsg.has_value())};
  builder.create<fir::CallOp>(loc, enableHandlers, ioArgs);
}

//===----------------------------------------------------------------------===//
// Data transfer helpers
//===----------------------------------------------------------------------===//

template <typename SEEK, typename A>
static bool hasIOControl(const A &stmt) {
  return hasX<SEEK>(stmt.controls);
}

template <typename SEEK, typename A>
static const auto *getIOControl(const A &stmt) {
  for (const auto &spec : stmt.controls)
    if (const auto *result = std::get_if<SEEK>(&spec.u))
      return result;
  return static_cast<const SEEK *>(nullptr);
}

/// Returns true iff the expression in the parse tree is not really a format but
/// rather a namelist group.
template <typename A>
static bool formatIsActuallyNamelist(const A &format) {
  if (auto *e = std::get_if<Fortran::parser::Expr>(&format.u)) {
    auto *expr = Fortran::semantics::GetExpr(*e);
    if (const Fortran::semantics::Symbol *y =
            Fortran::evaluate::UnwrapWholeSymbolDataRef(*expr))
      return y->has<Fortran::semantics::NamelistDetails>();
  }
  return false;
}

template <typename A>
static bool isDataTransferFormatted(const A &stmt) {
  if (stmt.format)
    return !formatIsActuallyNamelist(*stmt.format);
  return hasIOControl<Fortran::parser::Format>(stmt);
}
template <>
constexpr bool isDataTransferFormatted<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return true; // PRINT is always formatted
}

template <typename A>
static bool isDataTransferList(const A &stmt) {
  if (stmt.format)
    return std::holds_alternative<Fortran::parser::Star>(stmt.format->u);
  if (auto *mem = getIOControl<Fortran::parser::Format>(stmt))
    return std::holds_alternative<Fortran::parser::Star>(mem->u);
  return false;
}
template <>
bool isDataTransferList<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &stmt) {
  return std::holds_alternative<Fortran::parser::Star>(
      std::get<Fortran::parser::Format>(stmt.t).u);
}

template <typename A>
static bool isDataTransferInternal(const A &stmt) {
  if (stmt.iounit.has_value())
    return std::holds_alternative<Fortran::parser::Variable>(stmt.iounit->u);
  if (auto *unit = getIOControl<Fortran::parser::IoUnit>(stmt))
    return std::holds_alternative<Fortran::parser::Variable>(unit->u);
  return false;
}
template <>
constexpr bool isDataTransferInternal<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

/// If the variable `var` is an array or of a KIND other than the default
/// (normally 1), then a descriptor is required by the runtime IO API. This
/// condition holds even in F77 sources.
static std::optional<fir::ExtendedValue> getVariableBufferRequiredDescriptor(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::parser::Variable &var,
    Fortran::lower::StatementContext &stmtCtx) {
  fir::ExtendedValue varBox =
      converter.genExprBox(loc, var.typedExpr->v.value(), stmtCtx);
  fir::KindTy defCharKind = converter.getKindMap().defaultCharacterKind();
  mlir::Value varAddr = fir::getBase(varBox);
  if (fir::factory::CharacterExprHelper::getCharacterOrSequenceKind(
          varAddr.getType()) != defCharKind)
    return varBox;
  if (fir::factory::CharacterExprHelper::isArray(varAddr.getType()))
    return varBox;
  return std::nullopt;
}

template <typename A>
static std::optional<fir::ExtendedValue>
maybeGetInternalIODescriptor(Fortran::lower::AbstractConverter &converter,
                             mlir::Location loc, const A &stmt,
                             Fortran::lower::StatementContext &stmtCtx) {
  if (stmt.iounit.has_value())
    if (auto *var = std::get_if<Fortran::parser::Variable>(&stmt.iounit->u))
      return getVariableBufferRequiredDescriptor(converter, loc, *var, stmtCtx);
  if (auto *unit = getIOControl<Fortran::parser::IoUnit>(stmt))
    if (auto *var = std::get_if<Fortran::parser::Variable>(&unit->u))
      return getVariableBufferRequiredDescriptor(converter, loc, *var, stmtCtx);
  return std::nullopt;
}
template <>
inline std::optional<fir::ExtendedValue>
maybeGetInternalIODescriptor<Fortran::parser::PrintStmt>(
    Fortran::lower::AbstractConverter &, mlir::Location loc,
    const Fortran::parser::PrintStmt &, Fortran::lower::StatementContext &) {
  return std::nullopt;
}

template <typename A>
static bool isDataTransferAsynchronous(mlir::Location loc, const A &stmt) {
  if (auto *asynch =
          getIOControl<Fortran::parser::IoControlSpec::Asynchronous>(stmt)) {
    // FIXME: should contain a string of YES or NO
    TODO(loc, "asynchronous transfers not implemented in runtime");
  }
  return false;
}
template <>
bool isDataTransferAsynchronous<Fortran::parser::PrintStmt>(
    mlir::Location, const Fortran::parser::PrintStmt &) {
  return false;
}

template <typename A>
static bool isDataTransferNamelist(const A &stmt) {
  if (stmt.format)
    return formatIsActuallyNamelist(*stmt.format);
  return hasIOControl<Fortran::parser::Name>(stmt);
}
template <>
constexpr bool isDataTransferNamelist<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

/// Lowers a format statment that uses an assigned variable label reference as
/// a select operation to allow for run-time selection of the format statement.
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerReferenceAsStringSelect(Fortran::lower::AbstractConverter &converter,
                             mlir::Location loc,
                             const Fortran::lower::SomeExpr &expr,
                             mlir::Type strTy, mlir::Type lenTy,
                             Fortran::lower::StatementContext &stmtCtx) {
  // Create the requisite blocks to inline a selectOp.
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Block *startBlock = builder.getBlock();
  mlir::Block *endBlock = startBlock->splitBlock(builder.getInsertionPoint());
  mlir::Block *block = startBlock->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToEnd(block);

  llvm::SmallVector<int64_t> indexList;
  llvm::SmallVector<mlir::Block *> blockList;

  auto symbol = GetLastSymbol(&expr);
  Fortran::lower::pft::LabelSet labels;
  converter.lookupLabelSet(*symbol, labels);

  for (auto label : labels) {
    indexList.push_back(label);
    auto *eval = converter.lookupLabel(label);
    assert(eval && "Label is missing from the table");

    llvm::StringRef text = toStringRef(eval->position);
    mlir::Value stringRef;
    mlir::Value stringLen;
    if (eval->isA<Fortran::parser::FormatStmt>()) {
      assert(text.contains('(') && "FORMAT is unexpectedly ill-formed");
      // This is a format statement, so extract the spec from the text.
      std::tuple<mlir::Value, mlir::Value, mlir::Value> stringLit =
          lowerSourceTextAsStringLit(converter, loc, text, strTy, lenTy);
      stringRef = std::get<0>(stringLit);
      stringLen = std::get<1>(stringLit);
    } else {
      // This is not a format statement, so use null.
      stringRef = builder.createConvert(
          loc, strTy,
          builder.createIntegerConstant(loc, builder.getIndexType(), 0));
      stringLen = builder.createIntegerConstant(loc, lenTy, 0);
    }

    // Pass the format string reference and the string length out of the select
    // statement.
    llvm::SmallVector<mlir::Value> args = {stringRef, stringLen};
    builder.create<mlir::cf::BranchOp>(loc, endBlock, args);

    // Add block to the list of cases and make a new one.
    blockList.push_back(block);
    block = block->splitBlock(builder.getInsertionPoint());
    builder.setInsertionPointToEnd(block);
  }

  // Create the unit case which should result in an error.
  auto *unitBlock = block->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToEnd(unitBlock);
  fir::runtime::genReportFatalUserError(
      builder, loc,
      "Assigned format variable '" + symbol->name().ToString() +
          "' has not been assigned a valid format label");
  builder.create<fir::UnreachableOp>(loc);
  blockList.push_back(unitBlock);

  // Lower the selectOp.
  builder.setInsertionPointToEnd(startBlock);
  auto label = fir::getBase(converter.genExprValue(loc, &expr, stmtCtx));
  builder.create<fir::SelectOp>(loc, label, indexList, blockList);

  builder.setInsertionPointToEnd(endBlock);
  endBlock->addArgument(strTy, loc);
  endBlock->addArgument(lenTy, loc);

  // Handle and return the string reference and length selected by the selectOp.
  auto buff = endBlock->getArgument(0);
  auto len = endBlock->getArgument(1);

  return {buff, len, mlir::Value{}};
}

/// Generate a reference to a format string. There are four cases - a format
/// statement label, a character format expression, an integer that holds the
/// label of a format statement, and the * case. The first three are done here.
/// The * case is done elsewhere.
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
genFormat(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const Fortran::parser::Format &format, mlir::Type strTy,
          mlir::Type lenTy, Fortran::lower::StatementContext &stmtCtx) {
  if (const auto *label = std::get_if<Fortran::parser::Label>(&format.u)) {
    // format statement label
    auto eval = converter.lookupLabel(*label);
    assert(eval && "FORMAT not found in PROCEDURE");
    return lowerSourceTextAsStringLit(
        converter, loc, toStringRef(eval->position), strTy, lenTy);
  }
  const auto *pExpr = std::get_if<Fortran::parser::Expr>(&format.u);
  assert(pExpr && "missing format expression");
  auto e = Fortran::semantics::GetExpr(*pExpr);
  if (Fortran::semantics::ExprHasTypeCategory(
          *e, Fortran::common::TypeCategory::Character)) {
    // character expression
    if (e->Rank())
      // Array: return address(descriptor) and no length (and no kind value).
      return {fir::getBase(converter.genExprBox(loc, *e, stmtCtx)),
              mlir::Value{}, mlir::Value{}};
    // Scalar: return address(format) and format length (and no kind value).
    return lowerStringLit(converter, loc, stmtCtx, *pExpr, strTy, lenTy);
  }

  if (Fortran::semantics::ExprHasTypeCategory(
          *e, Fortran::common::TypeCategory::Integer) &&
      e->Rank() == 0 && Fortran::evaluate::UnwrapWholeSymbolDataRef(*e)) {
    // Treat as a scalar integer variable containing an ASSIGN label.
    return lowerReferenceAsStringSelect(converter, loc, *e, strTy, lenTy,
                                        stmtCtx);
  }

  // Legacy extension: it is possible that `*e` is not a scalar INTEGER
  // variable containing a label value. The output appears to be the source text
  // that initialized the variable? Needs more investigatation.
  TODO(loc, "io-control-spec contains a reference to a non-integer, "
            "non-scalar, or non-variable");
}

template <typename A>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getFormat(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const A &stmt, mlir::Type strTy, mlir::Type lenTy,
          Fortran ::lower::StatementContext &stmtCtx) {
  if (stmt.format && !formatIsActuallyNamelist(*stmt.format))
    return genFormat(converter, loc, *stmt.format, strTy, lenTy, stmtCtx);
  return genFormat(converter, loc, *getIOControl<Fortran::parser::Format>(stmt),
                   strTy, lenTy, stmtCtx);
}
template <>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getFormat<Fortran::parser::PrintStmt>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::parser::PrintStmt &stmt, mlir::Type strTy, mlir::Type lenTy,
    Fortran::lower::StatementContext &stmtCtx) {
  return genFormat(converter, loc, std::get<Fortran::parser::Format>(stmt.t),
                   strTy, lenTy, stmtCtx);
}

/// Get a buffer for an internal file data transfer.
template <typename A>
std::tuple<mlir::Value, mlir::Value>
getBuffer(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const A &stmt, mlir::Type strTy, mlir::Type lenTy,
          Fortran::lower::StatementContext &stmtCtx) {
  const Fortran::parser::IoUnit *iounit =
      stmt.iounit ? &*stmt.iounit : getIOControl<Fortran::parser::IoUnit>(stmt);
  if (iounit)
    if (auto *var = std::get_if<Fortran::parser::Variable>(&iounit->u))
      if (auto *expr = Fortran::semantics::GetExpr(*var))
        return genBuffer(converter, loc, *expr, strTy, lenTy, stmtCtx);
  llvm::report_fatal_error("failed to get IoUnit expr");
}

static mlir::Value genIOUnitNumber(Fortran::lower::AbstractConverter &converter,
                                   mlir::Location loc,
                                   const Fortran::lower::SomeExpr *iounit,
                                   mlir::Type ty, ConditionSpecInfo &csi,
                                   Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  auto rawUnit = fir::getBase(converter.genExprValue(loc, iounit, stmtCtx));
  unsigned rawUnitWidth =
      rawUnit.getType().cast<mlir::IntegerType>().getWidth();
  unsigned runtimeArgWidth = ty.cast<mlir::IntegerType>().getWidth();
  // The IO runtime supports `int` unit numbers, if the unit number may
  // overflow when passed to the IO runtime, check that the unit number is
  // in range before calling the BeginXXX.
  if (rawUnitWidth > runtimeArgWidth) {
    mlir::func::FuncOp check =
        rawUnitWidth <= 64
            ? getIORuntimeFunc<mkIOKey(CheckUnitNumberInRange64)>(loc, builder)
            : getIORuntimeFunc<mkIOKey(CheckUnitNumberInRange128)>(loc,
                                                                   builder);
    mlir::FunctionType funcTy = check.getFunctionType();
    llvm::SmallVector<mlir::Value> args;
    args.push_back(builder.createConvert(loc, funcTy.getInput(0), rawUnit));
    args.push_back(builder.createBool(loc, csi.hasErrorConditionSpec()));
    if (csi.ioMsg) {
      args.push_back(builder.createConvert(loc, funcTy.getInput(2),
                                           fir::getBase(*csi.ioMsg)));
      args.push_back(builder.createConvert(loc, funcTy.getInput(3),
                                           fir::getLen(*csi.ioMsg)));
    } else {
      args.push_back(builder.createNullConstant(loc, funcTy.getInput(2)));
      args.push_back(
          fir::factory::createZeroValue(builder, loc, funcTy.getInput(3)));
    }
    mlir::Value file = locToFilename(converter, loc, funcTy.getInput(4));
    mlir::Value line = locToLineNo(converter, loc, funcTy.getInput(5));
    args.push_back(file);
    args.push_back(line);
    auto checkCall = builder.create<fir::CallOp>(loc, check, args);
    if (csi.hasErrorConditionSpec()) {
      mlir::Value iostat = checkCall.getResult(0);
      mlir::Type iostatTy = iostat.getType();
      mlir::Value zero = fir::factory::createZeroValue(builder, loc, iostatTy);
      mlir::Value unitIsOK = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, iostat, zero);
      auto ifOp = builder.create<fir::IfOp>(loc, iostatTy, unitIsOK,
                                            /*withElseRegion=*/true);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      builder.create<fir::ResultOp>(loc, iostat);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      stmtCtx.pushScope();
      csi.bigUnitIfOp = ifOp;
    }
  }
  return builder.createConvert(loc, ty, rawUnit);
}

static mlir::Value genIOUnit(Fortran::lower::AbstractConverter &converter,
                             mlir::Location loc,
                             const Fortran::parser::IoUnit *iounit,
                             mlir::Type ty, ConditionSpecInfo &csi,
                             Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  if (iounit)
    if (auto *e = std::get_if<Fortran::parser::FileUnitNumber>(&iounit->u))
      return genIOUnitNumber(converter, loc, Fortran::semantics::GetExpr(*e),
                             ty, csi, stmtCtx);
  return builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIntegerAttr(ty, Fortran::runtime::io::DefaultUnit));
}

template <typename A>
static mlir::Value getIOUnit(Fortran::lower::AbstractConverter &converter,
                             mlir::Location loc, const A &stmt, mlir::Type ty,
                             ConditionSpecInfo &csi,
                             Fortran::lower::StatementContext &stmtCtx) {
  const Fortran::parser::IoUnit *iounit =
      stmt.iounit ? &*stmt.iounit : getIOControl<Fortran::parser::IoUnit>(stmt);
  return genIOUnit(converter, loc, iounit, ty, csi, stmtCtx);
}
//===----------------------------------------------------------------------===//
// Generators for each IO statement type.
//===----------------------------------------------------------------------===//

template <typename K, typename S>
static mlir::Value genBasicIOStmt(Fortran::lower::AbstractConverter &converter,
                                  const S &stmt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Location loc = converter.getCurrentLocation();
  ConditionSpecInfo csi = lowerErrorSpec(converter, loc, stmt.v);
  mlir::func::FuncOp beginFunc = getIORuntimeFunc<K>(loc, builder);
  mlir::FunctionType beginFuncTy = beginFunc.getFunctionType();
  mlir::Value unit = genIOUnitNumber(
      converter, loc, getExpr<Fortran::parser::FileUnitNumber>(stmt),
      beginFuncTy.getInput(0), csi, stmtCtx);
  mlir::Value un = builder.createConvert(loc, beginFuncTy.getInput(0), unit);
  mlir::Value file = locToFilename(converter, loc, beginFuncTy.getInput(1));
  mlir::Value line = locToLineNo(converter, loc, beginFuncTy.getInput(2));
  auto call = builder.create<fir::CallOp>(loc, beginFunc,
                                          mlir::ValueRange{un, file, line});
  mlir::Value cookie = call.getResult(0);
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  mlir::Value ok;
  auto insertPt = builder.saveInsertionPoint();
  threadSpecs(converter, loc, cookie, stmt.v, csi.hasErrorConditionSpec(), ok);
  builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, converter.getCurrentLocation(), cookie, csi,
                  stmtCtx);
}

mlir::Value Fortran::lower::genBackspaceStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::BackspaceStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginBackspace)>(converter, stmt);
}

mlir::Value Fortran::lower::genEndfileStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EndfileStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginEndfile)>(converter, stmt);
}

mlir::Value
Fortran::lower::genFlushStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::FlushStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginFlush)>(converter, stmt);
}

mlir::Value
Fortran::lower::genRewindStatement(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::parser::RewindStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginRewind)>(converter, stmt);
}

static mlir::Value
genNewunitSpec(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
               mlir::Value cookie,
               const std::list<Fortran::parser::ConnectSpec> &specList) {
  for (const auto &spec : specList)
    if (auto *newunit =
            std::get_if<Fortran::parser::ConnectSpec::Newunit>(&spec.u)) {
      Fortran::lower::StatementContext stmtCtx;
      fir::FirOpBuilder &builder = converter.getFirOpBuilder();
      mlir::func::FuncOp ioFunc =
          getIORuntimeFunc<mkIOKey(GetNewUnit)>(loc, builder);
      mlir::FunctionType ioFuncTy = ioFunc.getFunctionType();
      const auto *var = Fortran::semantics::GetExpr(newunit->v);
      mlir::Value addr = builder.createConvert(
          loc, ioFuncTy.getInput(1),
          fir::getBase(converter.genExprAddr(loc, var, stmtCtx)));
      auto kind = builder.createIntegerConstant(loc, ioFuncTy.getInput(2),
                                                var->GetType().value().kind());
      llvm::SmallVector<mlir::Value> ioArgs = {cookie, addr, kind};
      return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
    }
  llvm_unreachable("missing Newunit spec");
}

mlir::Value
Fortran::lower::genOpenStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::OpenStmt &stmt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::func::FuncOp beginFunc;
  llvm::SmallVector<mlir::Value> beginArgs;
  mlir::Location loc = converter.getCurrentLocation();
  ConditionSpecInfo csi = lowerErrorSpec(converter, loc, stmt.v);
  bool hasNewunitSpec = false;
  if (hasSpec<Fortran::parser::FileUnitNumber>(stmt)) {
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenUnit)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getFunctionType();
    mlir::Value unit = genIOUnitNumber(
        converter, loc, getExpr<Fortran::parser::FileUnitNumber>(stmt),
        beginFuncTy.getInput(0), csi, stmtCtx);
    beginArgs.push_back(unit);
    beginArgs.push_back(locToFilename(converter, loc, beginFuncTy.getInput(1)));
    beginArgs.push_back(locToLineNo(converter, loc, beginFuncTy.getInput(2)));
  } else {
    hasNewunitSpec = hasSpec<Fortran::parser::ConnectSpec::Newunit>(stmt);
    assert(hasNewunitSpec && "missing unit specifier");
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenNewUnit)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getFunctionType();
    beginArgs.push_back(locToFilename(converter, loc, beginFuncTy.getInput(0)));
    beginArgs.push_back(locToLineNo(converter, loc, beginFuncTy.getInput(1)));
  }
  auto cookie =
      builder.create<fir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  mlir::Value ok;
  auto insertPt = builder.saveInsertionPoint();
  threadSpecs(converter, loc, cookie, stmt.v, csi.hasErrorConditionSpec(), ok);
  if (hasNewunitSpec)
    genNewunitSpec(converter, loc, cookie, stmt.v);
  builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, loc, cookie, csi, stmtCtx);
}

mlir::Value
Fortran::lower::genCloseStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::CloseStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginClose)>(converter, stmt);
}

mlir::Value
Fortran::lower::genWaitStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::WaitStmt &stmt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Location loc = converter.getCurrentLocation();
  ConditionSpecInfo csi = lowerErrorSpec(converter, loc, stmt.v);
  bool hasId = hasSpec<Fortran::parser::IdExpr>(stmt);
  mlir::func::FuncOp beginFunc =
      hasId ? getIORuntimeFunc<mkIOKey(BeginWait)>(loc, builder)
            : getIORuntimeFunc<mkIOKey(BeginWaitAll)>(loc, builder);
  mlir::FunctionType beginFuncTy = beginFunc.getFunctionType();
  mlir::Value unit = genIOUnitNumber(
      converter, loc, getExpr<Fortran::parser::FileUnitNumber>(stmt),
      beginFuncTy.getInput(0), csi, stmtCtx);
  llvm::SmallVector<mlir::Value> args{unit};
  if (hasId) {
    mlir::Value id = fir::getBase(converter.genExprValue(
        loc, getExpr<Fortran::parser::IdExpr>(stmt), stmtCtx));
    args.push_back(builder.createConvert(loc, beginFuncTy.getInput(1), id));
    args.push_back(locToFilename(converter, loc, beginFuncTy.getInput(2)));
    args.push_back(locToLineNo(converter, loc, beginFuncTy.getInput(3)));
  } else {
    args.push_back(locToFilename(converter, loc, beginFuncTy.getInput(1)));
    args.push_back(locToLineNo(converter, loc, beginFuncTy.getInput(2)));
  }
  auto cookie = builder.create<fir::CallOp>(loc, beginFunc, args).getResult(0);
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  return genEndIO(converter, converter.getCurrentLocation(), cookie, csi,
                  stmtCtx);
}

//===----------------------------------------------------------------------===//
// Data transfer statements.
//
// There are several dimensions to the API with regard to data transfer
// statements that need to be considered.
//
//   - input (READ) vs. output (WRITE, PRINT)
//   - unformatted vs. formatted vs. list vs. namelist
//   - synchronous vs. asynchronous
//   - external vs. internal
//===----------------------------------------------------------------------===//

// Get the begin data transfer IO function to call for the given values.
template <bool isInput>
mlir::func::FuncOp
getBeginDataTransferFunc(mlir::Location loc, fir::FirOpBuilder &builder,
                         bool isFormatted, bool isListOrNml, bool isInternal,
                         bool isInternalWithDesc, bool isAsync) {
  if constexpr (isInput) {
    if (isFormatted || isListOrNml) {
      if (isInternal) {
        if (isInternalWithDesc) {
          if (isListOrNml)
            return getIORuntimeFunc<mkIOKey(BeginInternalArrayListInput)>(
                loc, builder);
          return getIORuntimeFunc<mkIOKey(BeginInternalArrayFormattedInput)>(
              loc, builder);
        }
        if (isListOrNml)
          return getIORuntimeFunc<mkIOKey(BeginInternalListInput)>(loc,
                                                                   builder);
        return getIORuntimeFunc<mkIOKey(BeginInternalFormattedInput)>(loc,
                                                                      builder);
      }
      if (isListOrNml)
        return getIORuntimeFunc<mkIOKey(BeginExternalListInput)>(loc, builder);
      return getIORuntimeFunc<mkIOKey(BeginExternalFormattedInput)>(loc,
                                                                    builder);
    }
    return getIORuntimeFunc<mkIOKey(BeginUnformattedInput)>(loc, builder);
  } else {
    if (isFormatted || isListOrNml) {
      if (isInternal) {
        if (isInternalWithDesc) {
          if (isListOrNml)
            return getIORuntimeFunc<mkIOKey(BeginInternalArrayListOutput)>(
                loc, builder);
          return getIORuntimeFunc<mkIOKey(BeginInternalArrayFormattedOutput)>(
              loc, builder);
        }
        if (isListOrNml)
          return getIORuntimeFunc<mkIOKey(BeginInternalListOutput)>(loc,
                                                                    builder);
        return getIORuntimeFunc<mkIOKey(BeginInternalFormattedOutput)>(loc,
                                                                       builder);
      }
      if (isListOrNml)
        return getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(loc, builder);
      return getIORuntimeFunc<mkIOKey(BeginExternalFormattedOutput)>(loc,
                                                                     builder);
    }
    return getIORuntimeFunc<mkIOKey(BeginUnformattedOutput)>(loc, builder);
  }
}

/// Generate the arguments of a begin data transfer statement call.
template <bool hasIOCtrl, typename A>
void genBeginDataTransferCallArgs(
    llvm::SmallVectorImpl<mlir::Value> &ioArgs,
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const A &stmt, mlir::FunctionType ioFuncTy, bool isFormatted,
    bool isListOrNml, [[maybe_unused]] bool isInternal,
    [[maybe_unused]] bool isAsync,
    const std::optional<fir::ExtendedValue> &descRef, ConditionSpecInfo &csi,
    Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto maybeGetFormatArgs = [&]() {
    if (!isFormatted || isListOrNml)
      return;
    std::tuple triple =
        getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                  ioFuncTy.getInput(ioArgs.size() + 1), stmtCtx);
    mlir::Value address = std::get<0>(triple);
    mlir::Value length = std::get<1>(triple);
    if (length) {
      // Scalar format: string arg + length arg; no format descriptor arg
      ioArgs.push_back(address); // format string
      ioArgs.push_back(length);  // format length
      ioArgs.push_back(
          builder.createNullConstant(loc, ioFuncTy.getInput(ioArgs.size())));
      return;
    }
    // Array format: no string arg, no length arg; format descriptor arg
    ioArgs.push_back(
        builder.createNullConstant(loc, ioFuncTy.getInput(ioArgs.size())));
    ioArgs.push_back(
        builder.createNullConstant(loc, ioFuncTy.getInput(ioArgs.size())));
    ioArgs.push_back( // format descriptor
        builder.createConvert(loc, ioFuncTy.getInput(ioArgs.size()), address));
  };
  if constexpr (hasIOCtrl) { // READ or WRITE
    if (isInternal) {
      // descriptor or scalar variable; maybe explicit format; scratch area
      if (descRef) {
        mlir::Value desc = builder.createBox(loc, *descRef);
        ioArgs.push_back(
            builder.createConvert(loc, ioFuncTy.getInput(ioArgs.size()), desc));
      } else {
        std::tuple<mlir::Value, mlir::Value> pair =
            getBuffer(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                      ioFuncTy.getInput(ioArgs.size() + 1), stmtCtx);
        ioArgs.push_back(std::get<0>(pair)); // scalar character variable
        ioArgs.push_back(std::get<1>(pair)); // character length
      }
      maybeGetFormatArgs();
      ioArgs.push_back( // internal scratch area buffer
          getDefaultScratch(builder, loc, ioFuncTy.getInput(ioArgs.size())));
      ioArgs.push_back( // buffer length
          getDefaultScratchLen(builder, loc, ioFuncTy.getInput(ioArgs.size())));
    } else { // external IO - maybe explicit format; unit
      if (isAsync)
        TODO(loc, "asynchronous");
      maybeGetFormatArgs();
      ioArgs.push_back(getIOUnit(converter, loc, stmt,
                                 ioFuncTy.getInput(ioArgs.size()), csi,
                                 stmtCtx));
    }
  } else { // PRINT - maybe explicit format; default unit
    maybeGetFormatArgs();
    ioArgs.push_back(builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(ioArgs.size()),
                                    Fortran::runtime::io::DefaultUnit)));
  }
  // File name and line number are always the last two arguments.
  ioArgs.push_back(
      locToFilename(converter, loc, ioFuncTy.getInput(ioArgs.size())));
  ioArgs.push_back(
      locToLineNo(converter, loc, ioFuncTy.getInput(ioArgs.size())));
}

template <bool isInput, bool hasIOCtrl = true, typename A>
static mlir::Value
genDataTransferStmt(Fortran::lower::AbstractConverter &converter,
                    const A &stmt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Location loc = converter.getCurrentLocation();
  const bool isFormatted = isDataTransferFormatted(stmt);
  const bool isList = isFormatted ? isDataTransferList(stmt) : false;
  const bool isInternal = isDataTransferInternal(stmt);
  std::optional<fir::ExtendedValue> descRef =
      isInternal ? maybeGetInternalIODescriptor(converter, loc, stmt, stmtCtx)
                 : std::nullopt;
  const bool isInternalWithDesc = descRef.has_value();
  const bool isAsync = isDataTransferAsynchronous(loc, stmt);
  const bool isNml = isDataTransferNamelist(stmt);

  // Generate an EnableHandlers call and remaining specifier calls.
  ConditionSpecInfo csi;
  if constexpr (hasIOCtrl) {
    csi = lowerErrorSpec(converter, loc, stmt.controls);
  }

  // Generate the begin data transfer function call.
  mlir::func::FuncOp ioFunc = getBeginDataTransferFunc<isInput>(
      loc, builder, isFormatted, isList || isNml, isInternal,
      isInternalWithDesc, isAsync);
  llvm::SmallVector<mlir::Value> ioArgs;
  genBeginDataTransferCallArgs<hasIOCtrl>(
      ioArgs, converter, loc, stmt, ioFunc.getFunctionType(), isFormatted,
      isList || isNml, isInternal, isAsync, descRef, csi, stmtCtx);
  mlir::Value cookie =
      builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);

  auto insertPt = builder.saveInsertionPoint();
  mlir::Value ok;
  if constexpr (hasIOCtrl) {
    genConditionHandlerCall(converter, loc, cookie, stmt.controls, csi);
    threadSpecs(converter, loc, cookie, stmt.controls,
                csi.hasErrorConditionSpec(), ok);
  }

  // Generate data transfer list calls.
  if constexpr (isInput) { // READ
    if (isNml)
      genNamelistIO(converter, cookie,
                    getIORuntimeFunc<mkIOKey(InputNamelist)>(loc, builder),
                    *getIOControl<Fortran::parser::Name>(stmt)->symbol,
                    csi.hasTransferConditionSpec(), ok, stmtCtx);
    else
      genInputItemList(converter, cookie, stmt.items, isFormatted,
                       csi.hasTransferConditionSpec(), ok, /*inLoop=*/false);
  } else if constexpr (std::is_same_v<A, Fortran::parser::WriteStmt>) {
    if (isNml)
      genNamelistIO(converter, cookie,
                    getIORuntimeFunc<mkIOKey(OutputNamelist)>(loc, builder),
                    *getIOControl<Fortran::parser::Name>(stmt)->symbol,
                    csi.hasTransferConditionSpec(), ok, stmtCtx);
    else
      genOutputItemList(converter, cookie, stmt.items, isFormatted,
                        csi.hasTransferConditionSpec(), ok,
                        /*inLoop=*/false);
  } else { // PRINT
    genOutputItemList(converter, cookie, std::get<1>(stmt.t), isFormatted,
                      csi.hasTransferConditionSpec(), ok,
                      /*inLoop=*/false);
  }

  builder.restoreInsertionPoint(insertPt);
  if constexpr (hasIOCtrl) {
    genIOReadSize(converter, loc, cookie, stmt.controls,
                  csi.hasErrorConditionSpec());
  }
  // Generate end statement call/s.
  mlir::Value result = genEndIO(converter, loc, cookie, csi, stmtCtx);
  stmtCtx.finalizeAndReset();
  return result;
}

void Fortran::lower::genPrintStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PrintStmt &stmt) {
  // PRINT does not take an io-control-spec. It only has a format specifier, so
  // it is a simplified case of WRITE.
  genDataTransferStmt</*isInput=*/false, /*ioCtrl=*/false>(converter, stmt);
}

mlir::Value
Fortran::lower::genWriteStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::WriteStmt &stmt) {
  return genDataTransferStmt</*isInput=*/false>(converter, stmt);
}

mlir::Value
Fortran::lower::genReadStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::ReadStmt &stmt) {
  return genDataTransferStmt</*isInput=*/true>(converter, stmt);
}

/// Get the file expression from the inquire spec list. Also return if the
/// expression is a file name.
static std::pair<const Fortran::lower::SomeExpr *, bool>
getInquireFileExpr(const std::list<Fortran::parser::InquireSpec> *stmt) {
  if (!stmt)
    return {nullptr, /*filename?=*/false};
  for (const Fortran::parser::InquireSpec &spec : *stmt) {
    if (auto *f = std::get_if<Fortran::parser::FileUnitNumber>(&spec.u))
      return {Fortran::semantics::GetExpr(*f), /*filename?=*/false};
    if (auto *f = std::get_if<Fortran::parser::FileNameExpr>(&spec.u))
      return {Fortran::semantics::GetExpr(*f), /*filename?=*/true};
  }
  // semantics should have already caught this condition
  llvm::report_fatal_error("inquire spec must have a file");
}

/// Generate calls to the four distinct INQUIRE subhandlers. An INQUIRE may
/// return values of type CHARACTER, INTEGER, or LOGICAL. There is one
/// additional special case for INQUIRE with both PENDING and ID specifiers.
template <typename A>
static mlir::Value genInquireSpec(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc, mlir::Value cookie,
                                  mlir::Value idExpr, const A &var,
                                  Fortran::lower::StatementContext &stmtCtx) {
  // default case: do nothing
  return {};
}
/// Specialization for CHARACTER.
template <>
mlir::Value genInquireSpec<Fortran::parser::InquireSpec::CharVar>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, mlir::Value idExpr,
    const Fortran::parser::InquireSpec::CharVar &var,
    Fortran::lower::StatementContext &stmtCtx) {
  // IOMSG is handled with exception conditions
  if (std::get<Fortran::parser::InquireSpec::CharVar::Kind>(var.t) ==
      Fortran::parser::InquireSpec::CharVar::Kind::Iomsg)
    return {};
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::func::FuncOp specFunc =
      getIORuntimeFunc<mkIOKey(InquireCharacter)>(loc, builder);
  mlir::FunctionType specFuncTy = specFunc.getFunctionType();
  const auto *varExpr = Fortran::semantics::GetExpr(
      std::get<Fortran::parser::ScalarDefaultCharVariable>(var.t));
  fir::ExtendedValue str = converter.genExprAddr(loc, varExpr, stmtCtx);
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, specFuncTy.getInput(0), cookie),
      builder.createIntegerConstant(
          loc, specFuncTy.getInput(1),
          Fortran::runtime::io::HashInquiryKeyword(std::string{
              Fortran::parser::InquireSpec::CharVar::EnumToString(
                  std::get<Fortran::parser::InquireSpec::CharVar::Kind>(var.t))}
                                                       .c_str())),
      builder.createConvert(loc, specFuncTy.getInput(2), fir::getBase(str)),
      builder.createConvert(loc, specFuncTy.getInput(3), fir::getLen(str))};
  return builder.create<fir::CallOp>(loc, specFunc, args).getResult(0);
}
/// Specialization for INTEGER.
template <>
mlir::Value genInquireSpec<Fortran::parser::InquireSpec::IntVar>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, mlir::Value idExpr,
    const Fortran::parser::InquireSpec::IntVar &var,
    Fortran::lower::StatementContext &stmtCtx) {
  // IOSTAT is handled with exception conditions
  if (std::get<Fortran::parser::InquireSpec::IntVar::Kind>(var.t) ==
      Fortran::parser::InquireSpec::IntVar::Kind::Iostat)
    return {};
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::func::FuncOp specFunc =
      getIORuntimeFunc<mkIOKey(InquireInteger64)>(loc, builder);
  mlir::FunctionType specFuncTy = specFunc.getFunctionType();
  const auto *varExpr = Fortran::semantics::GetExpr(
      std::get<Fortran::parser::ScalarIntVariable>(var.t));
  mlir::Value addr = fir::getBase(converter.genExprAddr(loc, varExpr, stmtCtx));
  mlir::Type eleTy = fir::dyn_cast_ptrEleTy(addr.getType());
  if (!eleTy)
    fir::emitFatalError(loc,
                        "internal error: expected a memory reference type");
  auto width = eleTy.cast<mlir::IntegerType>().getWidth();
  mlir::IndexType idxTy = builder.getIndexType();
  mlir::Value kind = builder.createIntegerConstant(loc, idxTy, width / 8);
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, specFuncTy.getInput(0), cookie),
      builder.createIntegerConstant(
          loc, specFuncTy.getInput(1),
          Fortran::runtime::io::HashInquiryKeyword(std::string{
              Fortran::parser::InquireSpec::IntVar::EnumToString(
                  std::get<Fortran::parser::InquireSpec::IntVar::Kind>(var.t))}
                                                       .c_str())),
      builder.createConvert(loc, specFuncTy.getInput(2), addr),
      builder.createConvert(loc, specFuncTy.getInput(3), kind)};
  return builder.create<fir::CallOp>(loc, specFunc, args).getResult(0);
}
/// Specialization for LOGICAL and (PENDING + ID).
template <>
mlir::Value genInquireSpec<Fortran::parser::InquireSpec::LogVar>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, mlir::Value idExpr,
    const Fortran::parser::InquireSpec::LogVar &var,
    Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto logVarKind = std::get<Fortran::parser::InquireSpec::LogVar::Kind>(var.t);
  bool pendId =
      idExpr &&
      logVarKind == Fortran::parser::InquireSpec::LogVar::Kind::Pending;
  mlir::func::FuncOp specFunc =
      pendId ? getIORuntimeFunc<mkIOKey(InquirePendingId)>(loc, builder)
             : getIORuntimeFunc<mkIOKey(InquireLogical)>(loc, builder);
  mlir::FunctionType specFuncTy = specFunc.getFunctionType();
  mlir::Value addr = fir::getBase(converter.genExprAddr(
      loc,
      Fortran::semantics::GetExpr(
          std::get<Fortran::parser::Scalar<
              Fortran::parser::Logical<Fortran::parser::Variable>>>(var.t)),
      stmtCtx));
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, specFuncTy.getInput(0), cookie)};
  if (pendId)
    args.push_back(builder.createConvert(loc, specFuncTy.getInput(1), idExpr));
  else
    args.push_back(builder.createIntegerConstant(
        loc, specFuncTy.getInput(1),
        Fortran::runtime::io::HashInquiryKeyword(std::string{
            Fortran::parser::InquireSpec::LogVar::EnumToString(logVarKind)}
                                                     .c_str())));
  args.push_back(builder.createConvert(loc, specFuncTy.getInput(2), addr));
  auto call = builder.create<fir::CallOp>(loc, specFunc, args);
  boolRefToLogical(loc, builder, addr);
  return call.getResult(0);
}

/// If there is an IdExpr in the list of inquire-specs, then lower it and return
/// the resulting Value. Otherwise, return null.
static mlir::Value
lowerIdExpr(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
            const std::list<Fortran::parser::InquireSpec> &ispecs,
            Fortran::lower::StatementContext &stmtCtx) {
  for (const Fortran::parser::InquireSpec &spec : ispecs)
    if (mlir::Value v = std::visit(
            Fortran::common::visitors{
                [&](const Fortran::parser::IdExpr &idExpr) {
                  return fir::getBase(converter.genExprValue(
                      loc, Fortran::semantics::GetExpr(idExpr), stmtCtx));
                },
                [](const auto &) { return mlir::Value{}; }},
            spec.u))
      return v;
  return {};
}

/// For each inquire-spec, build the appropriate call, threading the cookie.
static void threadInquire(Fortran::lower::AbstractConverter &converter,
                          mlir::Location loc, mlir::Value cookie,
                          const std::list<Fortran::parser::InquireSpec> &ispecs,
                          bool checkResult, mlir::Value &ok,
                          Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Value idExpr = lowerIdExpr(converter, loc, ispecs, stmtCtx);
  for (const Fortran::parser::InquireSpec &spec : ispecs) {
    makeNextConditionalOn(builder, loc, checkResult, ok);
    ok = std::visit(Fortran::common::visitors{[&](const auto &x) {
                      return genInquireSpec(converter, loc, cookie, idExpr, x,
                                            stmtCtx);
                    }},
                    spec.u);
  }
}

mlir::Value Fortran::lower::genInquireStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::InquireStmt &stmt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Location loc = converter.getCurrentLocation();
  mlir::func::FuncOp beginFunc;
  llvm::SmallVector<mlir::Value> beginArgs;
  const auto *list =
      std::get_if<std::list<Fortran::parser::InquireSpec>>(&stmt.u);
  auto exprPair = getInquireFileExpr(list);
  auto inquireFileUnit = [&]() -> bool {
    return exprPair.first && !exprPair.second;
  };
  auto inquireFileName = [&]() -> bool {
    return exprPair.first && exprPair.second;
  };

  ConditionSpecInfo csi =
      list ? lowerErrorSpec(converter, loc, *list) : ConditionSpecInfo{};

  // Make one of three BeginInquire calls.
  if (inquireFileUnit()) {
    // Inquire by unit -- [UNIT=]file-unit-number.
    beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireUnit)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getFunctionType();
    mlir::Value unit = genIOUnitNumber(converter, loc, exprPair.first,
                                       beginFuncTy.getInput(0), csi, stmtCtx);
    beginArgs = {unit, locToFilename(converter, loc, beginFuncTy.getInput(1)),
                 locToLineNo(converter, loc, beginFuncTy.getInput(2))};
  } else if (inquireFileName()) {
    // Inquire by file -- FILE=file-name-expr.
    beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireFile)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getFunctionType();
    fir::ExtendedValue file =
        converter.genExprAddr(loc, exprPair.first, stmtCtx);
    beginArgs = {
        builder.createConvert(loc, beginFuncTy.getInput(0), fir::getBase(file)),
        builder.createConvert(loc, beginFuncTy.getInput(1), fir::getLen(file)),
        locToFilename(converter, loc, beginFuncTy.getInput(2)),
        locToLineNo(converter, loc, beginFuncTy.getInput(3))};
  } else {
    // Inquire by output list -- IOLENGTH=scalar-int-variable.
    const auto *ioLength =
        std::get_if<Fortran::parser::InquireStmt::Iolength>(&stmt.u);
    assert(ioLength && "must have an IOLENGTH specifier");
    beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireIoLength)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getFunctionType();
    beginArgs = {locToFilename(converter, loc, beginFuncTy.getInput(0)),
                 locToLineNo(converter, loc, beginFuncTy.getInput(1))};
    auto cookie =
        builder.create<fir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
    mlir::Value ok;
    genOutputItemList(
        converter, cookie,
        std::get<std::list<Fortran::parser::OutputItem>>(ioLength->t),
        /*isFormatted=*/false, /*checkResult=*/false, ok, /*inLoop=*/false);
    auto *ioLengthVar = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::ScalarIntVariable>(ioLength->t));
    mlir::Value ioLengthVarAddr =
        fir::getBase(converter.genExprAddr(loc, ioLengthVar, stmtCtx));
    llvm::SmallVector<mlir::Value> args = {cookie};
    mlir::Value length =
        builder
            .create<fir::CallOp>(
                loc, getIORuntimeFunc<mkIOKey(GetIoLength)>(loc, builder), args)
            .getResult(0);
    mlir::Value length1 =
        builder.createConvert(loc, converter.genType(*ioLengthVar), length);
    builder.create<fir::StoreOp>(loc, length1, ioLengthVarAddr);
    return genEndIO(converter, loc, cookie, csi, stmtCtx);
  }

  // Common handling for inquire by unit or file.
  assert(list && "inquire-spec list must be present");
  auto cookie =
      builder.create<fir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
  genConditionHandlerCall(converter, loc, cookie, *list, csi);
  // Handle remaining arguments in specifier list.
  mlir::Value ok;
  auto insertPt = builder.saveInsertionPoint();
  threadInquire(converter, loc, cookie, *list, csi.hasErrorConditionSpec(), ok,
                stmtCtx);
  builder.restoreInsertionPoint(insertPt);
  // Generate end statement call.
  return genEndIO(converter, loc, cookie, csi, stmtCtx);
}
