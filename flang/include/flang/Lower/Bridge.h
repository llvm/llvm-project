//===-- Lower/Bridge.h -- main interface to lowering ------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_BRIDGE_H
#define FORTRAN_LOWER_BRIDGE_H

#include "flang/Common/Fortran.h"
#include "flang/Frontend/CodeGenOptions.h"
#include "flang/Frontend/TargetOptions.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/EnvironmentDefault.h"
#include "flang/Lower/LoweringOptions.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <set>

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
} // namespace common
namespace evaluate {
class IntrinsicProcTable;
class TargetCharacteristics;
} // namespace evaluate
namespace parser {
class AllCookedSources;
struct Program;
} // namespace parser
namespace semantics {
class SemanticsContext;
} // namespace semantics

namespace lower {

//===----------------------------------------------------------------------===//
// Lowering bridge
//===----------------------------------------------------------------------===//

/// The lowering bridge converts the front-end parse trees and semantics
/// checking residual to MLIR (FIR dialect) code.
class LoweringBridge {
public:
  /// Create a lowering bridge instance.
  static LoweringBridge
  create(mlir::MLIRContext &ctx,
         Fortran::semantics::SemanticsContext &semanticsContext,
         const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
         const Fortran::evaluate::IntrinsicProcTable &intrinsics,
         const Fortran::evaluate::TargetCharacteristics &targetCharacteristics,
         const Fortran::parser::AllCookedSources &allCooked,
         llvm::StringRef triple, fir::KindMapping &kindMap,
         const Fortran::lower::LoweringOptions &loweringOptions,
         const std::vector<Fortran::lower::EnvironmentDefault> &envDefaults,
         const Fortran::common::LanguageFeatureControl &languageFeatures,
         const llvm::TargetMachine &targetMachine,
         const Fortran::frontend::TargetOptions &targetOptions,
         const Fortran::frontend::CodeGenOptions &codeGenOptions) {
    return LoweringBridge(ctx, semanticsContext, defaultKinds, intrinsics,
                          targetCharacteristics, allCooked, triple, kindMap,
                          loweringOptions, envDefaults, languageFeatures,
                          targetMachine, targetOptions, codeGenOptions);
  }

  //===--------------------------------------------------------------------===//
  // Getters
  //===--------------------------------------------------------------------===//

  mlir::MLIRContext &getMLIRContext() { return context; }

  /// Get the ModuleOp. It can never be null, which is asserted in the ctor.
  mlir::ModuleOp getModule() { return *module; }
  mlir::ModuleOp getModuleAndRelease() { return module.release(); }

  const Fortran::common::IntrinsicTypeDefaultKinds &getDefaultKinds() const {
    return defaultKinds;
  }
  const Fortran::evaluate::IntrinsicProcTable &getIntrinsicTable() const {
    return intrinsics;
  }
  const Fortran::evaluate::TargetCharacteristics &
  getTargetCharacteristics() const {
    return targetCharacteristics;
  }
  const Fortran::parser::AllCookedSources *getCookedSource() const {
    return cooked;
  }

  /// Get the kind map.
  const fir::KindMapping &getKindMap() const { return kindMap; }

  const Fortran::lower::LoweringOptions &getLoweringOptions() const {
    return loweringOptions;
  }

  const std::vector<Fortran::lower::EnvironmentDefault> &
  getEnvironmentDefaults() const {
    return envDefaults;
  }

  const Fortran::common::LanguageFeatureControl &getLanguageFeatures() const {
    return languageFeatures;
  }

  /// Create a folding context. Careful: this is very expensive.
  Fortran::evaluate::FoldingContext createFoldingContext();

  Fortran::semantics::SemanticsContext &getSemanticsContext() const {
    return semanticsContext;
  }

  Fortran::lower::StatementContext &fctCtx() { return functionContext; }

  Fortran::lower::StatementContext &openAccCtx() { return openAccContext; }

  bool validModule() { return getModule(); }

  //===--------------------------------------------------------------------===//
  // Perform the creation of an mlir::ModuleOp
  //===--------------------------------------------------------------------===//

  /// Read in an MLIR input file rather than lowering Fortran sources.
  /// This is intended to be used for testing.
  void parseSourceFile(llvm::SourceMgr &);

  /// Cross the bridge from the Fortran parse-tree, etc. to MLIR dialects
  void lower(const Fortran::parser::Program &program,
             const Fortran::semantics::SemanticsContext &semanticsContext);

private:
  explicit LoweringBridge(
      mlir::MLIRContext &ctx,
      Fortran::semantics::SemanticsContext &semanticsContext,
      const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
      const Fortran::evaluate::IntrinsicProcTable &intrinsics,
      const Fortran::evaluate::TargetCharacteristics &targetCharacteristics,
      const Fortran::parser::AllCookedSources &cooked, llvm::StringRef triple,
      fir::KindMapping &kindMap,
      const Fortran::lower::LoweringOptions &loweringOptions,
      const std::vector<Fortran::lower::EnvironmentDefault> &envDefaults,
      const Fortran::common::LanguageFeatureControl &languageFeatures,
      const llvm::TargetMachine &targetMachine,
      const Fortran::frontend::TargetOptions &targetOptions,
      const Fortran::frontend::CodeGenOptions &codeGenOptions);
  LoweringBridge() = delete;
  LoweringBridge(const LoweringBridge &) = delete;

  Fortran::semantics::SemanticsContext &semanticsContext;
  Fortran::lower::StatementContext functionContext;
  Fortran::lower::StatementContext openAccContext;
  const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds;
  const Fortran::evaluate::IntrinsicProcTable &intrinsics;
  const Fortran::evaluate::TargetCharacteristics &targetCharacteristics;
  const Fortran::parser::AllCookedSources *cooked;
  mlir::MLIRContext &context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  fir::KindMapping &kindMap;
  const Fortran::lower::LoweringOptions &loweringOptions;
  const std::vector<Fortran::lower::EnvironmentDefault> &envDefaults;
  const Fortran::common::LanguageFeatureControl &languageFeatures;
  std::set<std::string> tempNames;
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_BRIDGE_H
