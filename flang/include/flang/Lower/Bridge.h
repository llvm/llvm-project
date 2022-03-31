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
#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/BuiltinOps.h"

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
} // namespace common
namespace evaluate {
class IntrinsicProcTable;
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
         const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
         const Fortran::evaluate::IntrinsicProcTable &intrinsics,
         const Fortran::parser::AllCookedSources &allCooked,
         llvm::StringRef triple, fir::KindMapping &kindMap) {
    return LoweringBridge(ctx, defaultKinds, intrinsics, allCooked, triple,
                          kindMap);
  }

  //===--------------------------------------------------------------------===//
  // Getters
  //===--------------------------------------------------------------------===//

  mlir::MLIRContext &getMLIRContext() { return context; }

  /// Get the ModuleOp. It can never be null, which is asserted in the ctor.
  mlir::ModuleOp &getModule() { return *module.get(); }

  const Fortran::common::IntrinsicTypeDefaultKinds &getDefaultKinds() const {
    return defaultKinds;
  }
  const Fortran::evaluate::IntrinsicProcTable &getIntrinsicTable() const {
    return intrinsics;
  }
  const Fortran::parser::AllCookedSources *getCookedSource() const {
    return cooked;
  }

  /// Get the kind map.
  const fir::KindMapping &getKindMap() const { return kindMap; }

  /// Create a folding context. Careful: this is very expensive.
  Fortran::evaluate::FoldingContext createFoldingContext() const;

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
      const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
      const Fortran::evaluate::IntrinsicProcTable &intrinsics,
      const Fortran::parser::AllCookedSources &cooked, llvm::StringRef triple,
      fir::KindMapping &kindMap);
  LoweringBridge() = delete;
  LoweringBridge(const LoweringBridge &) = delete;

  const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds;
  const Fortran::evaluate::IntrinsicProcTable &intrinsics;
  const Fortran::parser::AllCookedSources *cooked;
  mlir::MLIRContext &context;
  std::unique_ptr<mlir::ModuleOp> module;
  fir::KindMapping &kindMap;
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_BRIDGE_H
