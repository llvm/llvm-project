//===-- Lower/Bridge.h -- main interface to lowering ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements lowering. Convert Fortran source to
/// [MLIR](https://github.com/tensorflow/mlir).
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_BRIDGE_H
#define FORTRAN_LOWER_BRIDGE_H

#include "flang/Common/Fortran.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/BuiltinOps.h"

namespace fir {
struct NameUniquer;
}

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
  create(const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
         const Fortran::evaluate::IntrinsicProcTable &intrinsics,
         const Fortran::parser::AllCookedSources &allCooked) {
    return LoweringBridge{defaultKinds, intrinsics, allCooked};
  }

  //===--------------------------------------------------------------------===//
  // Getters
  //===--------------------------------------------------------------------===//

  mlir::MLIRContext &getMLIRContext() { return *context.get(); }
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
  void lower(const Fortran::parser::Program &program, fir::NameUniquer &uniquer,
             const Fortran::semantics::SemanticsContext &semanticsContext);

private:
  explicit LoweringBridge(
      const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
      const Fortran::evaluate::IntrinsicProcTable &intrinsics,
      const Fortran::parser::AllCookedSources &);
  LoweringBridge() = delete;
  LoweringBridge(const LoweringBridge &) = delete;

  const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds;
  const Fortran::evaluate::IntrinsicProcTable &intrinsics;
  const Fortran::parser::AllCookedSources *cooked;
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<mlir::ModuleOp> module;
  fir::KindMapping kindMap;
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_BRIDGE_H
