//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Optimizer/Transforms/
//  directory.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_TRANSFORMS_PASSES
#define FLANG_OPTIMIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AffineDialectPromotion : FunctionPass<"promote-to-affine"> {
  let summary = "Promotes `fir.{do_loop,if}` to `affine.{for,if}`.";
  let description = [{
    Convert fir operations which satisfy affine constraints to the affine
    dialect.

    `fir.do_loop` will be converted to `affine.for` if the loops inside the body
    can be converted and the indices for memory loads and stores satisfy
    `affine.apply` criteria for symbols and dimensions.

    `fir.if` will be converted to `affine.if` where possible. `affine.if`'s
    condition uses an integer set (==, >=) and an analysis is done to determine
    the fir condition's parent operations to construct the integer set.

    `fir.load` (`fir.store`) will be converted to `affine.load` (`affine.store`)
    where possible. This conversion includes adding a dummy `fir.convert` cast
    to adapt values of type `!fir.ref<!fir.array>` to `memref`. This is done
    because the affine dialect presently only understands the `memref` type.
  }];
  let constructor = "::fir::createPromoteToAffinePass()";
  let dependentDialects = [
    "fir::FIROpsDialect", "mlir::StandardOpsDialect", "mlir::AffineDialect"
  ];
}

def AffineDialectDemotion : FunctionPass<"demote-affine"> {
  let summary = "Converts `affine.{load,store}` back to fir operations";
  let description = [{
    Affine dialect's default lowering for loads and stores is different from
    fir as it uses the `memref` type. The `memref` type is not compatible with
    the Fortran runtime. Therefore, conversion of memory operations back to
    `fir.load` and `fir.store` with `!fir.ref<?>` types is required.
  }];
  let constructor = "::fir::createAffineDemotionPass()";
  let dependentDialects = [
    "fir::FIROpsDialect", "mlir::StandardOpsDialect", "mlir::AffineDialect"
  ];
}

def CharacterConversion : Pass<"character-conversion"> {
  let summary = "Convert CHARACTER entities with different KINDs";
  let description = [{
    Translates entities of one CHARACTER KIND to another.

    By default the translation is to naively zero-extend or truncate a code
    point to fit the destination size.
  }];
  let constructor = "::fir::createCharacterConversionPass()";
  let dependentDialects = [ "fir::FIROpsDialect" ];
  let options = [
    Option<"useRuntimeCalls", "use-runtime-calls",
           "std::string", /*default=*/"std::string{}",
           "Generate runtime calls to a named set of conversion routines. "
           "By default, the conversions may produce unexpected results.">
  ];
}

def ExternalNameConversion : Pass<"external-name-interop", "mlir::ModuleOp"> {
  let summary = "Convert name for external interoperability";
  let description = [{
    Demangle FIR internal name and mangle them for external interoperability.
  }];
  let constructor = "::fir::createExternalNameConversionPass()";
}

#endif // FLANG_OPTIMIZER_TRANSFORMS_PASSES
