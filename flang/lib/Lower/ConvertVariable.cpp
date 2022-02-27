//===-- ConvertVariable.cpp -- bridge to lower to MLIR --------------------===//
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

#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/BoxAnalyzer.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-variable"

/// Helper to lower a scalar expression using a specific symbol mapping.
static mlir::Value genScalarValue(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::SomeExpr &expr,
                                  Fortran::lower::SymMap &symMap,
                                  Fortran::lower::StatementContext &context) {
  // This does not use the AbstractConverter member function to override the
  // symbol mapping to be used expression lowering.
  return fir::getBase(Fortran::lower::createSomeExtendedExpression(
      loc, converter, expr, symMap, context));
}

//===----------------------------------------------------------------===//
// Local variables instantiation (not for alias)
//===----------------------------------------------------------------===//

/// Create a stack slot for a local variable. Precondition: the insertion
/// point of the builder must be in the entry block, which is currently being
/// constructed.
static mlir::Value createNewLocal(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::pft::Variable &var,
                                  mlir::Value preAlloc,
                                  llvm::ArrayRef<mlir::Value> shape = {},
                                  llvm::ArrayRef<mlir::Value> lenParams = {}) {
  if (preAlloc)
    return preAlloc;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string nm = Fortran::lower::mangle::mangleName(var.getSymbol());
  mlir::Type ty = converter.genType(var);
  const Fortran::semantics::Symbol &ultimateSymbol =
      var.getSymbol().GetUltimate();
  llvm::StringRef symNm = toStringRef(ultimateSymbol.name());
  bool isTarg = var.isTarget();
  // Let the builder do all the heavy lifting.
  return builder.allocateLocal(loc, ty, nm, symNm, shape, lenParams, isTarg);
}

/// Instantiate a local variable. Precondition: Each variable will be visited
/// such that if its properties depend on other variables, the variables upon
/// which its properties depend will already have been visited.
static void instantiateLocal(Fortran::lower::AbstractConverter &converter,
                             const Fortran::lower::pft::Variable &var,
                             Fortran::lower::SymMap &symMap) {
  assert(!var.isAlias());
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx);
}

/// Helper to decide if a dummy argument must be tracked in an BoxValue.
static bool lowerToBoxValue(const Fortran::semantics::Symbol &sym,
                            mlir::Value dummyArg) {
  // Only dummy arguments coming as fir.box can be tracked in an BoxValue.
  if (!dummyArg || !dummyArg.getType().isa<fir::BoxType>())
    return false;
  // Non contiguous arrays must be tracked in an BoxValue.
  if (sym.Rank() > 0 && !sym.attrs().test(Fortran::semantics::Attr::CONTIGUOUS))
    return true;
  // Assumed rank and optional fir.box cannot yet be read while lowering the
  // specifications.
  if (Fortran::evaluate::IsAssumedRank(sym) ||
      Fortran::semantics::IsOptional(sym))
    return true;
  // Polymorphic entity should be tracked through a fir.box that has the
  // dynamic type info.
  if (const Fortran::semantics::DeclTypeSpec *type = sym.GetType())
    if (type->IsPolymorphic())
      return true;
  return false;
}

/// Compute extent from lower and upper bound.
static mlir::Value computeExtent(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value lb, mlir::Value ub) {
  mlir::IndexType idxTy = builder.getIndexType();
  // Let the folder deal with the common `ub - <const> + 1` case.
  auto diff = builder.create<mlir::arith::SubIOp>(loc, idxTy, ub, lb);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  return builder.create<mlir::arith::AddIOp>(loc, idxTy, diff, one);
}

/// Lower explicit lower bounds into \p result. Does nothing if this is not an
/// array, or if the lower bounds are deferred, or all implicit or one.
static void lowerExplicitLowerBounds(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::BoxAnalyzer &box,
    llvm::SmallVectorImpl<mlir::Value> &result, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isArray() || box.lboundIsAllOnes())
    return;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  if (box.isStaticArray()) {
    for (int64_t lb : box.staticLBound())
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, lb));
    return;
  }
  for (const Fortran::semantics::ShapeSpec *spec : box.dynamicBound()) {
    if (auto low = spec->lbound().GetExplicit()) {
      auto expr = Fortran::lower::SomeExpr{*low};
      mlir::Value lb = builder.createConvert(
          loc, idxTy, genScalarValue(converter, loc, expr, symMap, stmtCtx));
      result.emplace_back(lb);
    } else if (!spec->lbound().isColon()) {
      // Implicit lower bound is 1 (Fortran 2018 section 8.5.8.3 point 3.)
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, 1));
    }
  }
  assert(result.empty() || result.size() == box.dynamicBound().size());
}

/// Lower explicit extents into \p result if this is an explicit-shape or
/// assumed-size array. Does nothing if this is not an explicit-shape or
/// assumed-size array.
static void lowerExplicitExtents(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc,
                                 const Fortran::lower::BoxAnalyzer &box,
                                 llvm::ArrayRef<mlir::Value> lowerBounds,
                                 llvm::SmallVectorImpl<mlir::Value> &result,
                                 Fortran::lower::SymMap &symMap,
                                 Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isArray())
    return;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  if (box.isStaticArray()) {
    for (int64_t extent : box.staticShape())
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, extent));
    return;
  }
  for (const auto &spec : llvm::enumerate(box.dynamicBound())) {
    if (auto up = spec.value()->ubound().GetExplicit()) {
      auto expr = Fortran::lower::SomeExpr{*up};
      mlir::Value ub = builder.createConvert(
          loc, idxTy, genScalarValue(converter, loc, expr, symMap, stmtCtx));
      if (lowerBounds.empty())
        result.emplace_back(ub);
      else
        result.emplace_back(
            computeExtent(builder, loc, lowerBounds[spec.index()], ub));
    } else if (spec.value()->ubound().isStar()) {
      // Assumed extent is undefined. Must be provided by user's code.
      result.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
    }
  }
  assert(result.empty() || result.size() == box.dynamicBound().size());
}

/// Treat negative values as undefined. Assumed size arrays will return -1 from
/// the front end for example. Using negative values can produce hard to find
/// bugs much further along in the compilation.
static mlir::Value genExtentValue(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Type idxTy,
                                  long frontEndExtent) {
  if (frontEndExtent >= 0)
    return builder.createIntegerConstant(loc, idxTy, frontEndExtent);
  return builder.create<fir::UndefOp>(loc, idxTy);
}

/// Lower specification expressions and attributes of variable \p var and
/// add it to the symbol map.
/// For global and aliases, the address must be pre-computed and provided
/// in \p preAlloc.
/// Dummy arguments must have already been mapped to mlir block arguments
/// their mapping may be updated here.
void Fortran::lower::mapSymbolAttributes(
    AbstractConverter &converter, const Fortran::lower::pft::Variable &var,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    mlir::Value preAlloc) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const mlir::Location loc = converter.genLocation(sym.name());
  mlir::IndexType idxTy = builder.getIndexType();
  const bool isDummy = Fortran::semantics::IsDummy(sym);
  const bool isResult = Fortran::semantics::IsFunctionResult(sym);
  const bool replace = isDummy || isResult;
  fir::factory::CharacterExprHelper charHelp{builder, loc};
  Fortran::lower::BoxAnalyzer ba;
  ba.analyze(sym);

  // First deal with pointers an allocatables, because their handling here
  // is the same regardless of their rank.
  if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
    // Get address of fir.box describing the entity.
    // global
    mlir::Value boxAlloc = preAlloc;
    // dummy or passed result
    if (!boxAlloc)
      if (Fortran::lower::SymbolBox symbox = symMap.lookupSymbol(sym))
        boxAlloc = symbox.getAddr();
    // local
    if (!boxAlloc)
      boxAlloc = createNewLocal(converter, loc, var, preAlloc);
    // Lower non deferred parameters.
    llvm::SmallVector<mlir::Value> nonDeferredLenParams;
    if (ba.isChar()) {
      TODO(loc, "mapSymbolAttributes allocatble or pointer char");
    } else if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType()) {
      if (const Fortran::semantics::DerivedTypeSpec *derived =
              declTy->AsDerived())
        if (Fortran::semantics::CountLenParameters(*derived) != 0)
          TODO(loc,
               "derived type allocatable or pointer with length parameters");
    }
    fir::MutableBoxValue box = Fortran::lower::createMutableBox(
        converter, loc, var, boxAlloc, nonDeferredLenParams);
    symMap.addAllocatableOrPointer(var.getSymbol(), box, replace);
    return;
  }

  if (isDummy) {
    mlir::Value dummyArg = symMap.lookupSymbol(sym).getAddr();
    if (lowerToBoxValue(sym, dummyArg)) {
      llvm::SmallVector<mlir::Value> lbounds;
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> explicitParams;
      // Lower lower bounds, explicit type parameters and explicit
      // extents if any.
      if (ba.isChar())
        TODO(loc, "lowerToBoxValue character");
      // TODO: derived type length parameters.
      lowerExplicitLowerBounds(converter, loc, ba, lbounds, symMap, stmtCtx);
      lowerExplicitExtents(converter, loc, ba, lbounds, extents, symMap,
                           stmtCtx);
      symMap.addBoxSymbol(sym, dummyArg, lbounds, explicitParams, extents,
                          replace);
      return;
    }
  }

  // For symbols reaching this point, all properties are constant and can be
  // read/computed already into ssa values.

  ba.match(
      //===--------------------------------------------------------------===//
      // Trivial case.
      //===--------------------------------------------------------------===//
      [&](const Fortran::lower::details::ScalarSym &) {
        if (isDummy) {
          // This is an argument.
          if (!symMap.lookupSymbol(sym))
            mlir::emitError(loc, "symbol \"")
                << toStringRef(sym.name()) << "\" must already be in map";
          return;
        } else if (isResult) {
          // Some Fortran results may be passed by argument (e.g. derived
          // types)
          if (symMap.lookupSymbol(sym))
            return;
        }
        // Otherwise, it's a local variable or function result.
        mlir::Value local = createNewLocal(converter, loc, var, preAlloc);
        symMap.addSymbol(sym, local);
      },

      //===--------------------------------------------------------------===//
      // The non-trivial cases are when we have an argument or local that has
      // a repetition value. Arguments might be passed as simple pointers and
      // need to be cast to a multi-dimensional array with constant bounds
      // (possibly with a missing column), bounds computed in the callee
      // (here), or with bounds from the caller (boxed somewhere else). Locals
      // have the same properties except they are never boxed arguments from
      // the caller and never having a missing column size.
      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::ScalarStaticChar &x) {
        // type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        if (replace) {
          Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(sym);
          std::pair<mlir::Value, mlir::Value> unboxchar =
              charHelp.createUnboxChar(symBox.getAddr());
          mlir::Value boxAddr = unboxchar.first;
          // Set/override LEN with a constant
          mlir::Value len = builder.createIntegerConstant(loc, idxTy, charLen);
          symMap.addCharSymbol(sym, boxAddr, len, true);
          return;
        }
        mlir::Value len = builder.createIntegerConstant(loc, idxTy, charLen);
        if (preAlloc) {
          symMap.addCharSymbol(sym, preAlloc, len);
          return;
        }
        mlir::Value local = createNewLocal(converter, loc, var, preAlloc);
        symMap.addCharSymbol(sym, local, len);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::ScalarDynamicChar &x) {
        TODO(loc, "ScalarDynamicChar variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArray &x) {
        // object shape is constant, not a character
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        mlir::Value addr = symMap.lookupSymbol(sym).getAddr();
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (int64_t i : x.shapes)
            shape.push_back(genExtentValue(builder, loc, idxTy, i));
          mlir::Value local =
              isDummy ? addr : createNewLocal(converter, loc, var, preAlloc);
          symMap.addSymbolWithShape(sym, local, shape, isDummy);
          return;
        }
        // If object is an array process the lower bound and extent values by
        // constructing constants and populating the lbounds and extents.
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(genExtentValue(builder, loc, idxTy, snd));
        }
        mlir::Value local =
            isDummy ? addr
                    : createNewLocal(converter, loc, var, preAlloc, extents);
        assert(isDummy || Fortran::lower::isExplicitShape(sym));
        symMap.addSymbolWithBounds(sym, local, extents, lbounds, isDummy);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArray &x) {
        TODO(loc, "DynamicArray variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArrayStaticChar &x) {
        TODO(loc, "StaticArrayStaticChar variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArrayDynamicChar &x) {
        TODO(loc, "StaticArrayDynamicChar variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArrayStaticChar &x) {
        TODO(loc, "DynamicArrayStaticChar variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArrayDynamicChar &x) {
        TODO(loc, "DynamicArrayDynamicChar variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::BoxAnalyzer::None &) {
        mlir::emitError(loc, "symbol analysis failed on ")
            << toStringRef(sym.name());
      });
}

void Fortran::lower::instantiateVariable(AbstractConverter &converter,
                                         const pft::Variable &var,
                                         SymMap &symMap) {
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const mlir::Location loc = converter.genLocation(sym.name());
  if (var.isAggregateStore()) {
    TODO(loc, "instantiateVariable AggregateStore");
  } else if (Fortran::semantics::FindCommonBlockContaining(
                 var.getSymbol().GetUltimate())) {
    TODO(loc, "instantiateVariable Common");
  } else if (var.isAlias()) {
    TODO(loc, "instantiateVariable Alias");
  } else if (var.isGlobal()) {
    TODO(loc, "instantiateVariable Global");
  } else {
    instantiateLocal(converter, var, symMap);
  }
}

void Fortran::lower::mapCallInterfaceSymbols(
    AbstractConverter &converter, const Fortran::lower::CallerInterface &caller,
    SymMap &symMap) {
  const Fortran::semantics::Symbol &result = caller.getResultSymbol();
  for (Fortran::lower::pft::Variable var :
       Fortran::lower::pft::buildFuncResultDependencyList(result)) {
    if (var.isAggregateStore()) {
      instantiateVariable(converter, var, symMap);
    } else {
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      const auto *hostDetails =
          sym.detailsIf<Fortran::semantics::HostAssocDetails>();
      if (hostDetails && !var.isModuleVariable()) {
        // The callee is an internal procedure `A` whose result properties
        // depend on host variables. The caller may be the host, or another
        // internal procedure `B` contained in the same host.  In the first
        // case, the host symbol is obviously mapped, in the second case, it
        // must also be mapped because
        // HostAssociations::internalProcedureBindings that was called when
        // lowering `B` will have mapped all host symbols of captured variables
        // to the tuple argument containing the composite of all host associated
        // variables, whether or not the host symbol is actually referred to in
        // `B`. Hence it is possible to simply lookup the variable associated to
        // the host symbol without having to go back to the tuple argument.
        Fortran::lower::SymbolBox hostValue =
            symMap.lookupSymbol(hostDetails->symbol());
        assert(hostValue && "callee host symbol must be mapped on caller side");
        symMap.addSymbol(sym, hostValue.toExtendedValue());
        // The SymbolBox associated to the host symbols is complete, skip
        // instantiateVariable that would try to allocate a new storage.
        continue;
      }
      if (Fortran::semantics::IsDummy(sym) && sym.owner() == result.owner()) {
        // Get the argument for the dummy argument symbols of the current call.
        symMap.addSymbol(sym, caller.getArgumentValue(sym));
        // All the properties of the dummy variable may not come from the actual
        // argument, let instantiateVariable handle this.
      }
      // If this is neither a host associated or dummy symbol, it must be a
      // module or common block variable to satisfy specification expression
      // requirements in 10.1.11, instantiateVariable will get its address and
      // properties.
      instantiateVariable(converter, var, symMap);
    }
  }
}
