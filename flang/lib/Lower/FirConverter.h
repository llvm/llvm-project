//===-- FirConverter.h ----------------------------------------------------===//
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

#ifndef FORTRAN_LOWER_FIRCONVERTER_H
#define FORTRAN_LOWER_FIRCONVERTER_H

#include "flang/Common/Fortran.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/IterationSpace.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/PFTDefs.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/iostat.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>

namespace Fortran::lower {

class FirConverter : public Fortran::lower::AbstractConverter {
public:
  explicit FirConverter(Fortran::lower::LoweringBridge &bridge)
      : Fortran::lower::AbstractConverter(bridge.getLoweringOptions()),
        bridge{bridge}, foldingContext{bridge.createFoldingContext()} {}
  virtual ~FirConverter() = default;

  void run(Fortran::lower::pft::Program &pft);

  /// The core of the conversion: take an evaluation and generate FIR for it.
  /// The generation for each individual element of PFT is done via a specific
  /// genFIR function (see below).
  /// This function will automatically call the genFIR function for the type
  /// of the PFT construct.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              bool unstructuredContext = true);

private:
  // All core Fortran constructs:

  void genFIR(const Fortran::parser::AllocateStmt &);
  void genFIR(const Fortran::parser::ArithmeticIfStmt &);
  void genFIR(const Fortran::parser::AssignedGotoStmt &);
  void genFIR(const Fortran::parser::AssignmentStmt &);
  void genFIR(const Fortran::parser::AssignStmt &);
  void genFIR(const Fortran::parser::AssociateConstruct &);
  void genFIR(const Fortran::parser::BackspaceStmt &);
  void genFIR(const Fortran::parser::BlockConstruct &);
  void genFIR(const Fortran::parser::CallStmt &);
  void genFIR(const Fortran::parser::CaseConstruct &);
  void genFIR(const Fortran::parser::ChangeTeamConstruct &);
  void genFIR(const Fortran::parser::ChangeTeamStmt &);
  void genFIR(const Fortran::parser::CloseStmt &);
  void genFIR(const Fortran::parser::CompilerDirective &);
  void genFIR(const Fortran::parser::ComputedGotoStmt &);
  void genFIR(const Fortran::parser::ConcurrentHeader &);
  void genFIR(const Fortran::parser::CriticalConstruct &);
  void genFIR(const Fortran::parser::CriticalStmt &);
  void genFIR(const Fortran::parser::CycleStmt &);
  void genFIR(const Fortran::parser::DeallocateStmt &);
  void genFIR(const Fortran::parser::DoConstruct &);
  void genFIR(const Fortran::parser::ElsewhereStmt &);
  void genFIR(const Fortran::parser::EndChangeTeamStmt &);
  void genFIR(const Fortran::parser::EndCriticalStmt &);
  void genFIR(const Fortran::parser::EndfileStmt &);
  void genFIR(const Fortran::parser::EndForallStmt &);
  void genFIR(const Fortran::parser::EndWhereStmt &);
  void genFIR(const Fortran::parser::EventPostStmt &);
  void genFIR(const Fortran::parser::EventWaitStmt &);
  void genFIR(const Fortran::parser::ExitStmt &);
  void genFIR(const Fortran::parser::FailImageStmt &);
  void genFIR(const Fortran::parser::FlushStmt &);
  void genFIR(const Fortran::parser::ForallAssignmentStmt &);
  void genFIR(const Fortran::parser::ForallConstruct &);
  void genFIR(const Fortran::parser::ForallConstructStmt &);
  void genFIR(const Fortran::parser::ForallStmt &);
  void genFIR(const Fortran::parser::FormatStmt &);
  void genFIR(const Fortran::parser::FormTeamStmt &);
  void genFIR(const Fortran::parser::GotoStmt &);
  void genFIR(const Fortran::parser::IfConstruct &);
  void genFIR(const Fortran::parser::InquireStmt &);
  void genFIR(const Fortran::parser::LockStmt &);
  void genFIR(const Fortran::parser::MaskedElsewhereStmt &);
  void genFIR(const Fortran::parser::NullifyStmt &);
  void genFIR(const Fortran::parser::OpenACCConstruct &);
  void genFIR(const Fortran::parser::OpenACCDeclarativeConstruct &);
  void genFIR(const Fortran::parser::OpenACCRoutineConstruct &);
  void genFIR(const Fortran::parser::OpenMPConstruct &);
  void genFIR(const Fortran::parser::OpenMPDeclarativeConstruct &);
  void genFIR(const Fortran::parser::OpenStmt &);
  void genFIR(const Fortran::parser::PauseStmt &);
  void genFIR(const Fortran::parser::PointerAssignmentStmt &);
  void genFIR(const Fortran::parser::PrintStmt &);
  void genFIR(const Fortran::parser::ReadStmt &);
  void genFIR(const Fortran::parser::ReturnStmt &);
  void genFIR(const Fortran::parser::RewindStmt &);
  void genFIR(const Fortran::parser::SelectCaseStmt &);
  void genFIR(const Fortran::parser::SelectRankCaseStmt &);
  void genFIR(const Fortran::parser::SelectRankConstruct &);
  void genFIR(const Fortran::parser::SelectRankStmt &);
  void genFIR(const Fortran::parser::SelectTypeConstruct &);
  void genFIR(const Fortran::parser::StopStmt &);
  void genFIR(const Fortran::parser::SyncAllStmt &);
  void genFIR(const Fortran::parser::SyncImagesStmt &);
  void genFIR(const Fortran::parser::SyncMemoryStmt &);
  void genFIR(const Fortran::parser::SyncTeamStmt &);
  void genFIR(const Fortran::parser::UnlockStmt &);
  void genFIR(const Fortran::parser::WaitStmt &);
  void genFIR(const Fortran::parser::WhereBodyConstruct &);
  void genFIR(const Fortran::parser::WhereConstruct &);
  void genFIR(const Fortran::parser::WhereConstruct::Elsewhere &);
  void genFIR(const Fortran::parser::WhereConstruct::MaskedElsewhere &);
  void genFIR(const Fortran::parser::WhereConstructStmt &);
  void genFIR(const Fortran::parser::WhereStmt &);
  void genFIR(const Fortran::parser::WriteStmt &);

  // Nop statements - No code, or code is generated at the construct level.
  // But note that the genFIR call immediately below that wraps one of these
  // calls does block management, possibly starting a new block, and possibly
  // generating a branch to end a block. So these calls may still be required
  // for that functionality.
  void genFIR(const Fortran::parser::AssociateStmt &) {}       // nop
  void genFIR(const Fortran::parser::BlockStmt &) {}           // nop
  void genFIR(const Fortran::parser::CaseStmt &) {}            // nop
  void genFIR(const Fortran::parser::ContinueStmt &) {}        // nop
  void genFIR(const Fortran::parser::ElseIfStmt &) {}          // nop
  void genFIR(const Fortran::parser::ElseStmt &) {}            // nop
  void genFIR(const Fortran::parser::EndAssociateStmt &) {}    // nop
  void genFIR(const Fortran::parser::EndBlockStmt &) {}        // nop
  void genFIR(const Fortran::parser::EndDoStmt &) {}           // nop
  void genFIR(const Fortran::parser::EndFunctionStmt &) {}     // nop
  void genFIR(const Fortran::parser::EndIfStmt &) {}           // nop
  void genFIR(const Fortran::parser::EndMpSubprogramStmt &) {} // nop
  void genFIR(const Fortran::parser::EndProgramStmt &) {}      // nop
  void genFIR(const Fortran::parser::EndSelectStmt &) {}       // nop
  void genFIR(const Fortran::parser::EndSubroutineStmt &) {}   // nop
  void genFIR(const Fortran::parser::EntryStmt &) {}           // nop
  void genFIR(const Fortran::parser::IfStmt &) {}              // nop
  void genFIR(const Fortran::parser::IfThenStmt &) {}          // nop
  void genFIR(const Fortran::parser::NonLabelDoStmt &) {}      // nop
  void genFIR(const Fortran::parser::OmpEndLoopDirective &) {} // nop
  void genFIR(const Fortran::parser::SelectTypeStmt &) {}      // nop
  void genFIR(const Fortran::parser::TypeGuardStmt &) {}       // nop

public:
  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

  mlir::Value getSymbolAddress(Fortran::lower::SymbolRef sym) override final {
    return lookupSymbol(sym).getAddr();
  }

  fir::ExtendedValue
  symBoxToExtendedValue(const Fortran::lower::SymbolBox &symBox);

  fir::ExtendedValue
  getSymbolExtendedValue(const Fortran::semantics::Symbol &sym,
                         Fortran::lower::SymMap *symMap) override final;

  mlir::Value impliedDoBinding(llvm::StringRef name) override final;

  void copySymbolBinding(Fortran::lower::SymbolRef src,
                         Fortran::lower::SymbolRef target) override final {
    localSymbols.copySymbolBinding(src, target);
  }

  bool bindIfNewSymbol(Fortran::lower::SymbolRef sym,
                       const fir::ExtendedValue &exval);

  void bindSymbol(Fortran::lower::SymbolRef sym,
                  const fir::ExtendedValue &exval) override final {
    addSymbol(sym, exval, /*forced=*/true);
  }

  void
  overrideExprValues(const Fortran::lower::ExprToValueMap *map) override final {
    exprValueOverrides = map;
  }

  const Fortran::lower::ExprToValueMap *getExprOverrides() override final {
    return exprValueOverrides;
  }

  bool lookupLabelSet(Fortran::lower::SymbolRef sym,
                      Fortran::lower::pft::LabelSet &labelSet) override final;

  Fortran::lower::pft::Evaluation *
  lookupLabel(Fortran::lower::pft::Label label) override final {
    Fortran::lower::pft::FunctionLikeUnit &owningProc =
        *getEval().getOwningProcedure();
    return owningProc.labelEvaluationMap.lookup(label);
  }

  fir::ExtendedValue
  genExprAddr(const Fortran::lower::SomeExpr &expr,
              Fortran::lower::StatementContext &context,
              mlir::Location *locPtr = nullptr) override final;

  fir::ExtendedValue
  genExprValue(const Fortran::lower::SomeExpr &expr,
               Fortran::lower::StatementContext &context,
               mlir::Location *locPtr = nullptr) override final;

  fir::ExtendedValue
  genExprBox(mlir::Location loc, const Fortran::lower::SomeExpr &expr,
             Fortran::lower::StatementContext &stmtCtx) override final;

  Fortran::evaluate::FoldingContext &getFoldingContext() override final {
    return foldingContext;
  }

  mlir::Type genType(const Fortran::lower::SomeExpr &expr) override final {
    return Fortran::lower::translateSomeExprToFIRType(*this, expr);
  }

  mlir::Type genType(const Fortran::lower::pft::Variable &var) override final {
    return Fortran::lower::translateVariableToFIRType(*this, var);
  }

  mlir::Type genType(Fortran::lower::SymbolRef sym) override final {
    return Fortran::lower::translateSymbolToFIRType(*this, sym);
  }

  mlir::Type
  genType(Fortran::common::TypeCategory tc, int kind,
          llvm::ArrayRef<std::int64_t> lenParameters) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(), tc, kind,
                                      lenParameters);
  }

  mlir::Type
  genType(const Fortran::semantics::DerivedTypeSpec &tySpec) override final {
    return Fortran::lower::translateDerivedTypeToFIRType(*this, tySpec);
  }

  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    return Fortran::lower::getFIRType(
        &getMLIRContext(), tc, bridge.getDefaultKinds().GetDefaultKind(tc),
        std::nullopt);
  }

  bool isPresentShallowLookup(Fortran::semantics::Symbol &sym) override final {
    return bool(shallowLookupSymbol(sym));
  }

  bool createHostAssociateVarClone(
      const Fortran::semantics::Symbol &sym) override final;

  void createHostAssociateVarCloneDealloc(
      const Fortran::semantics::Symbol &sym) override final;

  void copyHostAssociateVar(
      const Fortran::semantics::Symbol &sym,
      mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr) override final;

  //===--------------------------------------------------------------------===//
  // Utility methods
  //===--------------------------------------------------------------------===//

  void collectSymbolSet(
      Fortran::lower::pft::Evaluation &eval,
      llvm::SetVector<const Fortran::semantics::Symbol *> &symbolSet,
      Fortran::semantics::Symbol::Flag flag, bool collectSymbols,
      bool checkHostAssociatedSymbols) override final;

  mlir::Location getCurrentLocation() override final { return toLocation(); }

  /// Generate a dummy location.
  mlir::Location genUnknownLocation() override final {
    // Note: builder may not be instantiated yet
    return mlir::UnknownLoc::get(&getMLIRContext());
  }

  mlir::Location
  genLocation(const Fortran::parser::CharBlock &block) override final;

  const Fortran::semantics::Scope &getCurrentScope() override final {
    return bridge.getSemanticsContext().FindScope(currentPosition);
  }

  fir::FirOpBuilder &getFirOpBuilder() override final { return *builder; }

  mlir::ModuleOp &getModuleOp() override final { return bridge.getModule(); }

  mlir::MLIRContext &getMLIRContext() override final {
    return bridge.getMLIRContext();
  }
  std::string
  mangleName(const Fortran::semantics::Symbol &symbol) override final {
    return Fortran::lower::mangle::mangleName(
        symbol, scopeBlockIdMap, /*keepExternalInScope=*/false,
        getLoweringOptions().getUnderscoring());
  }
  std::string mangleName(
      const Fortran::semantics::DerivedTypeSpec &derivedType) override final {
    return Fortran::lower::mangle::mangleName(derivedType, scopeBlockIdMap);
  }
  std::string mangleName(std::string &name) override final {
    return Fortran::lower::mangle::mangleName(name, getCurrentScope(),
                                              scopeBlockIdMap);
  }
  std::string getRecordTypeFieldName(
      const Fortran::semantics::Symbol &component) override final {
    return Fortran::lower::mangle::getRecordTypeFieldName(component,
                                                          scopeBlockIdMap);
  }
  const fir::KindMapping &getKindMap() override final {
    return bridge.getKindMap();
  }

  Fortran::lower::StatementContext &getFctCtx() override final;

  mlir::Value hostAssocTupleValue() override final { return hostAssocTuple; }

  /// Record a binding for the ssa-value of the tuple for this function.
  void bindHostAssocTuple(mlir::Value val) override final {
    assert(!hostAssocTuple && val);
    hostAssocTuple = val;
  }

  void registerTypeInfo(mlir::Location loc,
                        Fortran::lower::SymbolRef typeInfoSym,
                        const Fortran::semantics::DerivedTypeSpec &typeSpec,
                        fir::RecordType type) override final {
    typeInfoConverter.registerTypeInfo(*this, loc, typeInfoSym, typeSpec, type);
  }

  llvm::StringRef
  getUniqueLitName(mlir::Location loc,
                   std::unique_ptr<Fortran::lower::SomeExpr> expr,
                   mlir::Type eleTy) override final;

private:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  /// Helper classes

  /// Information for generating a structured or unstructured increment loop.
  struct IncrementLoopInfo {
    template <typename T>
    explicit IncrementLoopInfo(Fortran::semantics::Symbol &sym, const T &lower,
                               const T &upper, const std::optional<T> &step,
                               bool isUnordered = false)
        : loopVariableSym{&sym}, lowerExpr{Fortran::semantics::GetExpr(lower)},
          upperExpr{Fortran::semantics::GetExpr(upper)},
          stepExpr{Fortran::semantics::GetExpr(step)},
          isUnordered{isUnordered} {}

    IncrementLoopInfo(IncrementLoopInfo &&) = default;
    IncrementLoopInfo &operator=(IncrementLoopInfo &&x) = default;

    bool isStructured() const { return !headerBlock; }

    mlir::Type getLoopVariableType() const {
      assert(loopVariable && "must be set");
      return fir::unwrapRefType(loopVariable.getType());
    }

    bool hasLocalitySpecs() const {
      return !localSymList.empty() || !localInitSymList.empty() ||
             !sharedSymList.empty();
    }

    // Data members common to both structured and unstructured loops.
    const Fortran::semantics::Symbol *loopVariableSym;
    const Fortran::lower::SomeExpr *lowerExpr;
    const Fortran::lower::SomeExpr *upperExpr;
    const Fortran::lower::SomeExpr *stepExpr;
    const Fortran::lower::SomeExpr *maskExpr = nullptr;
    bool isUnordered; // do concurrent, forall
    llvm::SmallVector<const Fortran::semantics::Symbol *> localSymList;
    llvm::SmallVector<const Fortran::semantics::Symbol *> localInitSymList;
    llvm::SmallVector<const Fortran::semantics::Symbol *> sharedSymList;
    mlir::Value loopVariable = nullptr;

    // Data members for structured loops.
    fir::DoLoopOp doLoop = nullptr;

    // Data members for unstructured loops.
    bool hasRealControl = false;
    mlir::Value tripVariable = nullptr;
    mlir::Value stepVariable = nullptr;
    mlir::Block *headerBlock = nullptr; // loop entry and test block
    mlir::Block *maskBlock = nullptr;   // concurrent loop mask block
    mlir::Block *bodyBlock = nullptr;   // first loop body block
    mlir::Block *exitBlock = nullptr;   // loop exit target block
  };

  using IncrementLoopNestInfo = llvm::SmallVector<IncrementLoopInfo, 8>;

  /// Information to support stack management, object deallocation, and
  /// object finalization at early and normal construct exits.
  struct ConstructContext {
    explicit ConstructContext(Fortran::lower::pft::Evaluation &eval,
                              Fortran::lower::StatementContext &stmtCtx)
        : eval{eval}, stmtCtx{stmtCtx} {}

    Fortran::lower::pft::Evaluation &eval;     // construct eval
    Fortran::lower::StatementContext &stmtCtx; // construct exit code
  };

  /// Helper class to generate the runtime type info global data and the
  /// fir.type_info operations that contain the dipatch tables (if any).
  /// The type info global data is required to describe the derived type to the
  /// runtime so that it can operate over it.
  /// It must be ensured these operations will be generated for every derived
  /// type lowered in the current translated unit. However, these operations
  /// cannot be generated before FuncOp have been created for functions since
  /// the initializers may take their address (e.g for type bound procedures).
  /// This class allows registering all the required type info while it is not
  /// possible to create GlobalOp/TypeInfoOp, and to generate this data afte
  /// function lowering.
  class TypeInfoConverter {
    /// Store the location and symbols of derived type info to be generated.
    /// The location of the derived type instantiation is also stored because
    /// runtime type descriptor symbols are compiler generated and cannot be
    /// mapped to user code on their own.
    struct TypeInfo {
      Fortran::semantics::SymbolRef symbol;
      const Fortran::semantics::DerivedTypeSpec &typeSpec;
      fir::RecordType type;
      mlir::Location loc;
    };

  public:
    void registerTypeInfo(Fortran::lower::AbstractConverter &converter,
                          mlir::Location loc,
                          Fortran::semantics::SymbolRef typeInfoSym,
                          const Fortran::semantics::DerivedTypeSpec &typeSpec,
                          fir::RecordType type);
    void createTypeInfo(Fortran::lower::AbstractConverter &converter);

  private:
    void createTypeInfoOpAndGlobal(Fortran::lower::AbstractConverter &converter,
                                   const TypeInfo &info);
    void createTypeInfoOp(Fortran::lower::AbstractConverter &converter,
                          const TypeInfo &info);

    /// Store the front-end data that will be required to generate the type info
    /// for the derived types that have been converted to fir.type<>.
    llvm::SmallVector<TypeInfo> registeredTypeInfo;
    /// Create derived type info immediately without storing the
    /// symbol in registeredTypeInfo.
    bool skipRegistration = false;
    /// Track symbols symbols processed during and after the registration
    /// to avoid infinite loops between type conversions and global variable
    /// creation.
    llvm::SmallSetVector<Fortran::semantics::SymbolRef, 32> seen;
  };

  void declareFunction(Fortran::lower::pft::FunctionLikeUnit &funit);

  const Fortran::semantics::Scope &
  getSymbolHostScope(const Fortran::semantics::Symbol &sym);

  void collectHostAssociatedVariables(
      Fortran::lower::pft::FunctionLikeUnit &funit,
      llvm::SetVector<const Fortran::semantics::Symbol *> &escapees);

  //===--------------------------------------------------------------------===//
  // Helper member functions
  //===--------------------------------------------------------------------===//

  mlir::Value createFIRExpr(mlir::Location loc,
                            const Fortran::lower::SomeExpr *expr,
                            Fortran::lower::StatementContext &stmtCtx) {
    return fir::getBase(genExprValue(*expr, stmtCtx, &loc));
  }

  Fortran::lower::SymbolBox
  lookupSymbol(const Fortran::semantics::Symbol &sym,
               Fortran::lower::SymMap *symMap = nullptr);

  /// Find the symbol in the inner-most level of the local map or return null.
  Fortran::lower::SymbolBox
  shallowLookupSymbol(const Fortran::semantics::Symbol &sym) {
    if (Fortran::lower::SymbolBox v = localSymbols.shallowLookupSymbol(sym))
      return v;
    return {};
  }

  /// Find the symbol in one level up of symbol map such as for host-association
  /// in OpenMP code or return null.
  Fortran::lower::SymbolBox
  lookupOneLevelUpSymbol(const Fortran::semantics::Symbol &sym) {
    if (Fortran::lower::SymbolBox v = localSymbols.lookupOneLevelUpSymbol(sym))
      return v;
    return {};
  }

  bool addSymbol(const Fortran::semantics::SymbolRef sym,
                 fir::ExtendedValue val, bool forced = false);

  bool mapBlockArgToDummyOrResult(const Fortran::semantics::SymbolRef sym,
                                  mlir::Value val, bool forced = false);

  mlir::Value genLoopVariableAddress(mlir::Location loc,
                                     const Fortran::semantics::Symbol &sym,
                                     bool isUnordered);

  static bool isNumericScalarCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Integer ||
           cat == Fortran::common::TypeCategory::Real ||
           cat == Fortran::common::TypeCategory::Complex ||
           cat == Fortran::common::TypeCategory::Logical;
  }
  static bool isLogicalCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Logical;
  }
  static bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Character;
  }
  static bool isDerivedCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Derived;
  }

  mlir::Block *insertBlock(mlir::Block *block);

  Fortran::lower::pft::Evaluation &evalOfLabel(Fortran::parser::Label label);

  void genBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    builder->create<mlir::cf::BranchOp>(toLocation(), targetBlock);
  }

  void genConditionalBranch(mlir::Value cond, mlir::Block *trueTarget,
                            mlir::Block *falseTarget);
  void genConditionalBranch(mlir::Value cond,
                            Fortran::lower::pft::Evaluation *trueTarget,
                            Fortran::lower::pft::Evaluation *falseTarget) {
    genConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }

  void genConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                            mlir::Block *trueTarget, mlir::Block *falseTarget);
  void genConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                            Fortran::lower::pft::Evaluation *trueTarget,
                            Fortran::lower::pft::Evaluation *falseTarget);

  Fortran::lower::pft::Evaluation *
  getActiveAncestor(const Fortran::lower::pft::Evaluation &eval);

  bool hasExitCode(const Fortran::lower::pft::Evaluation &targetEval);

  void
  genConstructExitBranch(const Fortran::lower::pft::Evaluation &targetEval);

  void genMultiwayBranch(mlir::Value selector,
                         llvm::SmallVector<int64_t> valueList,
                         llvm::SmallVector<Fortran::parser::Label> labelList,
                         const Fortran::lower::pft::Evaluation &defaultEval,
                         mlir::Block *errorBlock = nullptr);

  void pushActiveConstruct(Fortran::lower::pft::Evaluation &eval,
                           Fortran::lower::StatementContext &stmtCtx) {
    activeConstructStack.push_back(ConstructContext{eval, stmtCtx});
    eval.activeConstruct = true;
  }

  void popActiveConstruct() {
    assert(!activeConstructStack.empty() && "invalid active construct stack");
    activeConstructStack.back().eval.activeConstruct = false;
    activeConstructStack.pop_back();
  }

  //===--------------------------------------------------------------------===//
  // Termination of symbolically referenced execution units
  //===--------------------------------------------------------------------===//

  /// END of program
  ///
  /// Generate the cleanup block before the program exits
  void genExitRoutine() {
    if (blockIsUnterminated())
      builder->create<mlir::func::ReturnOp>(toLocation());
  }

  /// END of procedure-like constructs
  ///
  void genReturnSymbol(const Fortran::semantics::Symbol &functionSymbol);

  /// Get the return value of a call to \p symbol, which is a subroutine entry
  /// point that has alternative return specifiers.
  const mlir::Value
  getAltReturnResult(const Fortran::semantics::Symbol &symbol) {
    assert(Fortran::semantics::HasAlternateReturns(symbol) &&
           "subroutine does not have alternate returns");
    return getSymbolAddress(symbol);
  }

  void genFIRProcedureExit(Fortran::lower::pft::FunctionLikeUnit &funit,
                           const Fortran::semantics::Symbol &symbol);

  //
  // Statements that have control-flow semantics
  //

  /// Generate an If[Then]Stmt condition or its negation.
  template <typename A>
  mlir::Value genIfCondition(const A *stmt, bool negate = false) {
    mlir::Location loc = toLocation();
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value condExpr = createFIRExpr(
        loc,
        Fortran::semantics::GetExpr(
            std::get<Fortran::parser::ScalarLogicalExpr>(stmt->t)),
        stmtCtx);
    stmtCtx.finalizeAndReset();
    mlir::Value cond =
        builder->createConvert(loc, builder->getI1Type(), condExpr);
    if (negate)
      cond = builder->create<mlir::arith::XOrIOp>(
          loc, cond, builder->createIntegerConstant(loc, cond.getType(), 1));
    return cond;
  }

  mlir::func::FuncOp getFunc(llvm::StringRef name, mlir::FunctionType ty);

  void genFIRIncrementLoopBegin(IncrementLoopNestInfo &incrementLoopNestInfo);
  void genFIRIncrementLoopEnd(IncrementLoopNestInfo &incrementLoopNestInfo);
  IncrementLoopNestInfo getConcurrentControl(
      const Fortran::parser::ConcurrentHeader &header,
      const std::list<Fortran::parser::LocalitySpec> &localityList = {});

  void handleLocalitySpecs(const IncrementLoopInfo &info);

  mlir::Value genControlValue(const Fortran::lower::SomeExpr *expr,
                              const IncrementLoopInfo &info,
                              bool *isConst = nullptr);

  template <typename A>
  void genNestedStatement(const Fortran::parser::Statement<A> &stmt) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }

  void forceControlVariableBinding(const Fortran::semantics::Symbol *sym,
                                   mlir::Value inducVar);

  template <typename A> void prepareExplicitSpace(const A &forall) {
    if (!explicitIterSpace.isActive())
      analyzeExplicitSpace(forall);
    localSymbols.pushScope();
    explicitIterSpace.enter();
  }

  /// Cleanup all the FORALL context information when we exit.
  void cleanupExplicitSpace() {
    explicitIterSpace.leave();
    localSymbols.popScope();
  }

  void genForallNest(const Fortran::parser::ConcurrentHeader &header);

  fir::ExtendedValue
  genAssociateSelector(const Fortran::lower::SomeExpr &selector,
                       Fortran::lower::StatementContext &stmtCtx);

  template <typename A>
  void genIoConditionBranches(Fortran::lower::pft::Evaluation &eval,
                              const A &specList, mlir::Value iostat) {
    if (!iostat)
      return;

    Fortran::parser::Label endLabel{};
    Fortran::parser::Label eorLabel{};
    Fortran::parser::Label errLabel{};
    bool hasIostat{};
    for (const auto &spec : specList) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::EndLabel &label) {
                endLabel = label.v;
              },
              [&](const Fortran::parser::EorLabel &label) {
                eorLabel = label.v;
              },
              [&](const Fortran::parser::ErrLabel &label) {
                errLabel = label.v;
              },
              [&](const Fortran::parser::StatVariable &) { hasIostat = true; },
              [](const auto &) {}},
          spec.u);
    }
    if (!endLabel && !eorLabel && !errLabel)
      return;

    // An ERR specifier branch is taken on any positive error value rather than
    // some single specific value. If ERR and IOSTAT specifiers are given and
    // END and EOR specifiers are allowed, the latter two specifiers must have
    // explicit branch targets to allow the ERR branch to be implemented as a
    // default/else target. A label=0 target for an absent END or EOR specifier
    // indicates that these specifiers have a fallthrough target. END and EOR
    // specifiers may appear on READ and WAIT statements.
    bool allSpecifiersRequired = errLabel && hasIostat &&
                                 (eval.isA<Fortran::parser::ReadStmt>() ||
                                  eval.isA<Fortran::parser::WaitStmt>());
    mlir::Value selector =
        builder->createConvert(toLocation(), builder->getIndexType(), iostat);
    llvm::SmallVector<int64_t> valueList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    if (eorLabel || allSpecifiersRequired) {
      valueList.push_back(Fortran::runtime::io::IostatEor);
      labelList.push_back(eorLabel ? eorLabel : 0);
    }
    if (endLabel || allSpecifiersRequired) {
      valueList.push_back(Fortran::runtime::io::IostatEnd);
      labelList.push_back(endLabel ? endLabel : 0);
    }
    if (errLabel) {
      // Must be last. Value 0 is interpreted as any positive value, or
      // equivalently as any value other than 0, IostatEor, or IostatEnd.
      valueList.push_back(0);
      labelList.push_back(errLabel);
    }
    genMultiwayBranch(selector, valueList, labelList, eval.nonNopSuccessor());
  }

  fir::ExtendedValue
  genInitializerExprValue(const Fortran::lower::SomeExpr &expr,
                          Fortran::lower::StatementContext &stmtCtx) {
    return Fortran::lower::createSomeInitializerExpression(
        toLocation(), *this, expr, localSymbols, stmtCtx);
  }

  /// Return true if the current context is a conditionalized and implied
  /// iteration space.
  bool implicitIterationSpace() { return !implicitIterSpace.empty(); }

  /// Return true if context is currently an explicit iteration space. A scalar
  /// assignment expression may be contextually within a user-defined iteration
  /// space, transforming it into an array expression.
  bool explicitIterationSpace() { return explicitIterSpace.isActive(); }

  void genArrayAssignment(
      const Fortran::evaluate::Assignment &assign,
      Fortran::lower::StatementContext &localStmtCtx,
      std::optional<llvm::SmallVector<mlir::Value>> lbounds = std::nullopt,
      std::optional<llvm::SmallVector<mlir::Value>> ubounds = std::nullopt);

#if !defined(NDEBUG)
  static bool isFuncResultDesignator(const Fortran::lower::SomeExpr &expr) {
    const Fortran::semantics::Symbol *sym =
        Fortran::evaluate::GetFirstSymbol(expr);
    return sym && sym->IsFuncResult();
  }
#endif

  fir::MutableBoxValue
  genExprMutableBox(mlir::Location loc,
                    const Fortran::lower::SomeExpr &expr) override final;

  mlir::Value createLboundArray(llvm::ArrayRef<mlir::Value> lbounds,
                                mlir::Location loc);

  mlir::Value createBoundArray(llvm::ArrayRef<mlir::Value> lbounds,
                               llvm::ArrayRef<mlir::Value> ubounds,
                               mlir::Location loc);

  void genPointerAssignment(
      mlir::Location loc, const Fortran::evaluate::Assignment &assign,
      const Fortran::evaluate::Assignment::BoundsSpec &lbExprs);
  void genPointerAssignment(
      mlir::Location loc, const Fortran::evaluate::Assignment &assign,
      const Fortran::evaluate::Assignment::BoundsRemapping &boundExprs);

  hlfir::Entity genImplicitConvert(const Fortran::evaluate::Assignment &assign,
                                   hlfir::Entity rhs, bool preserveLowerBounds,
                                   Fortran::lower::StatementContext &stmtCtx);

  static void
  genCleanUpInRegionIfAny(mlir::Location loc, fir::FirOpBuilder &builder,
                          mlir::Region &region,
                          Fortran::lower::StatementContext &context);

  bool firstDummyIsPointerOrAllocatable(
      const Fortran::evaluate::ProcedureRef &userDefinedAssignment);

  void genDataAssignment(
      const Fortran::evaluate::Assignment &assign,
      const Fortran::evaluate::ProcedureRef *userDefinedAssignment);

  void genAssignment(const Fortran::evaluate::Assignment &assign);

  // Is the insertion point of the builder directly or indirectly set
  // inside any operation of type "Op"?
  template <typename... Op> bool isInsideOp() const {
    mlir::Block *block = builder->getInsertionBlock();
    mlir::Operation *op = block ? block->getParentOp() : nullptr;
    while (op) {
      if (mlir::isa<Op...>(op))
        return true;
      op = op->getParentOp();
    }
    return false;
  }

  bool isInsideHlfirForallOrWhere() const {
    return isInsideOp<hlfir::ForallOp, hlfir::WhereOp>();
  }

  bool isInsideHlfirWhere() const { return isInsideOp<hlfir::WhereOp>(); }

  void lowerWhereMaskToHlfir(mlir::Location loc,
                             const Fortran::semantics::SomeExpr *maskExpr);

  void mapDummiesAndResults(Fortran::lower::pft::FunctionLikeUnit &funit,
                            const Fortran::lower::CalleeInterface &callee);

  void instantiateVar(const Fortran::lower::pft::Variable &var,
                      Fortran::lower::AggregateStoreMap &storeMap);

  void manageFPEnvironment(Fortran::lower::pft::FunctionLikeUnit &funit);

  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit);

  void
  createEmptyBlocks(std::list<Fortran::lower::pft::Evaluation> &evaluationList);

  /// Return the predicate: "current block does not have a terminator branch".
  bool blockIsUnterminated() {
    mlir::Block *currentBlock = builder->getBlock();
    return currentBlock->empty() ||
           !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>();
  }

  void startBlock(mlir::Block *newBlock);

  /// Conditionally switch code insertion to a new block.
  void maybeStartBlock(mlir::Block *newBlock) {
    if (newBlock)
      startBlock(newBlock);
  }

  void eraseDeadCodeAndBlocks(mlir::RewriterBase &rewriter,
                              llvm::MutableArrayRef<mlir::Region> regions);

  void endNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit);

  void createGlobalOutsideOfFunctionLowering(
      const std::function<void()> &createGlobals);

  void lowerBlockData(Fortran::lower::pft::BlockDataUnit &bdunit);

  /// Create fir::Global for all the common blocks that appear in the program.
  void
  lowerCommonBlocks(const Fortran::semantics::CommonBlockList &commonBlocks) {
    createGlobalOutsideOfFunctionLowering(
        [&]() { Fortran::lower::defineCommonBlocks(*this, commonBlocks); });
  }

  void createIntrinsicModuleDefinitions(Fortran::lower::pft::Program &pft);

  void lowerFunc(Fortran::lower::pft::FunctionLikeUnit &funit);
  void lowerModuleDeclScope(Fortran::lower::pft::ModuleLikeUnit &mod);
  void lowerMod(Fortran::lower::pft::ModuleLikeUnit &mod);

  void setCurrentPosition(const Fortran::parser::CharBlock &position) {
    if (position != Fortran::parser::CharBlock{})
      currentPosition = position;
  }

  /// Set current position at the location of \p parseTreeNode. Note that the
  /// position is updated automatically when visiting statements, but not when
  /// entering higher level nodes like constructs or procedures. This helper is
  /// intended to cover the latter cases.
  template <typename A> void setCurrentPositionAt(const A &parseTreeNode) {
    setCurrentPosition(Fortran::parser::FindSourceLocation(parseTreeNode));
  }

  //===--------------------------------------------------------------------===//
  // Utility methods
  //===--------------------------------------------------------------------===//

  /// Convert a parser CharBlock to a Location
  mlir::Location toLocation(const Fortran::parser::CharBlock &cb) {
    return genLocation(cb);
  }

  mlir::Location toLocation() { return toLocation(currentPosition); }
  void setCurrentEval(Fortran::lower::pft::Evaluation &eval) {
    evalPtr = &eval;
  }

  Fortran::lower::pft::Evaluation &getEval() {
    assert(evalPtr);
    return *evalPtr;
  }

  std::optional<Fortran::evaluate::Shape>
  getShape(const Fortran::lower::SomeExpr &expr) {
    return Fortran::evaluate::GetShape(foldingContext, expr);
  }

  //===--------------------------------------------------------------------===//
  // Analysis on a nested explicit iteration space.
  //===--------------------------------------------------------------------===//

  template <bool LHS = false, typename A>
  void analyzeExplicitSpace(const Fortran::evaluate::Expr<A> &e) {
    explicitIterSpace.exprBase(&e, LHS);
  }

  void analyzeExplicitSpace(const Fortran::evaluate::Assignment *assign);

  void analyzeExplicitSpace(const Fortran::parser::AssignmentStmt &s) {
    analyzeExplicitSpace(s.typedAssignment->v.operator->());
  }

  void analyzeExplicitSpace(const Fortran::parser::ConcurrentHeader &header);
  void analyzeExplicitSpace(const Fortran::parser::ForallAssignmentStmt &stmt) {
    std::visit([&](const auto &s) { analyzeExplicitSpace(s); }, stmt.u);
  }

  void analyzeExplicitSpace(const Fortran::parser::ForallConstruct &forall);
  void analyzeExplicitSpace(const Fortran::parser::ForallConstructStmt &forall);
  void analyzeExplicitSpace(const Fortran::parser::ForallStmt &forall);
  void analyzeExplicitSpace(const Fortran::parser::MaskedElsewhereStmt &stmt);
  void analyzeExplicitSpace(const Fortran::parser::PointerAssignmentStmt &s) {
    analyzeExplicitSpace(s.typedAssignment->v.operator->());
  }

  void analyzeExplicitSpace(const Fortran::parser::WhereBodyConstruct &body);
  void analyzeExplicitSpace(const Fortran::parser::WhereConstruct &c);
  void analyzeExplicitSpace(
      const Fortran::parser::WhereConstruct::Elsewhere *ew);
  void analyzeExplicitSpace(
      const Fortran::parser::WhereConstruct::MaskedElsewhere &ew);
  void analyzeExplicitSpace(const Fortran::parser::WhereConstructStmt &ws);
  void analyzeExplicitSpace(const Fortran::parser::WhereStmt &stmt);

  void analyzeExplicitSpacePop() { explicitIterSpace.popLevel(); }

  void addMaskVariable(Fortran::lower::FrontEndExpr exp);

  void createRuntimeTypeInfoGlobals() {}

  bool lowerToHighLevelFIR() const {
    return bridge.getLoweringOptions().getLowerToHighLevelFIR();
  }

  std::string getConstantExprManglePrefix(mlir::Location loc,
                                          const Fortran::lower::SomeExpr &expr,
                                          mlir::Type eleTy);

  void finalizeOpenACCLowering();
  void finalizeOpenMPLowering(
      const Fortran::semantics::Symbol *globalOmpRequiresSymbol);

  //===--------------------------------------------------------------------===//

  Fortran::lower::LoweringBridge &bridge;
  Fortran::evaluate::FoldingContext foldingContext;
  fir::FirOpBuilder *builder = nullptr;
  Fortran::lower::pft::Evaluation *evalPtr = nullptr;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;
  TypeInfoConverter typeInfoConverter;

  // Stack to manage object deallocation and finalization at construct exits.
  llvm::SmallVector<ConstructContext> activeConstructStack;

  /// BLOCK name mangling component map
  int blockId = 0;
  Fortran::lower::mangle::ScopeBlockIdMap scopeBlockIdMap;

  /// FORALL statement/construct context
  Fortran::lower::ExplicitIterSpace explicitIterSpace;

  /// WHERE statement/construct mask expression stack
  Fortran::lower::ImplicitIterSpace implicitIterSpace;

  /// Tuple of host associated variables
  mlir::Value hostAssocTuple;

  /// A map of unique names for constant expressions.
  /// The names are used for representing the constant expressions
  /// with global constant initialized objects.
  /// The names are usually prefixed by a mangling string based
  /// on the element type of the constant expression, but the element
  /// type is not used as a key into the map (so the assumption is that
  /// the equivalent constant expressions are prefixed using the same
  /// element type).
  llvm::DenseMap<const Fortran::lower::SomeExpr *, std::string> literalNamesMap;

  /// Storage for Constant expressions used as keys for literalNamesMap.
  llvm::SmallVector<std::unique_ptr<Fortran::lower::SomeExpr>>
      literalExprsStorage;

  /// A counter for uniquing names in `literalNamesMap`.
  std::uint64_t uniqueLitId = 0;

  /// Deferred OpenACC routine attachment.
  Fortran::lower::AccRoutineInfoMappingList accRoutineInfos;

  /// Whether an OpenMP target region or declare target function/subroutine
  /// intended for device offloading has been detected
  bool ompDeviceCodeFound = false;

  const Fortran::lower::ExprToValueMap *exprValueOverrides{nullptr};
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRCONVERTER_H
