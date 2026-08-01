//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE
#define CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

def LoopOpInterface : OpInterface<"LoopOpInterface", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  DeclareOpInterfaceMethods<LoopLikeOpInterface>
]> {
  let description = [{
    Contains helper functions to query properties and perform transformations
    on a loop.
  }];
  let cppNamespace = "::cir";

  let methods = [
    InterfaceMethod<[{
        Returns the loop's conditional region.
      }],
      /*retTy=*/"mlir::Region &",
      /*methodName=*/"getCond"
    >,
    InterfaceMethod<[{
        Returns the loop's body region.
      }],
      /*retTy=*/"mlir::Region &",
      /*methodName=*/"getBody"
    >,
    InterfaceMethod<[{
        Returns a pointer to the loop's step region or nullptr.
      }],
      /*retTy=*/"mlir::Region *",
      /*methodName=*/"maybeGetStep",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return nullptr;"
    >,
    InterfaceMethod<[{
        Returns a pointer to the loop's per-iteration cleanup region, or
        nullptr if the loop has no (non-empty) cleanup region. When present,
        this region runs on every per-iteration exit edge (condition-false
        exit, end-of-iteration, break/continue, and EH unwinding when the
        cleanup kind includes EH).
      }],
      /*retTy=*/"mlir::Region *",
      /*methodName=*/"maybeGetCleanup",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return nullptr;"
    >,
    InterfaceMethod<[{
        Returns the cleanup-kind attribute for the loop's per-iteration
        cleanup region, or a null attribute if the loop has no cleanup kind
        (e.g. do-while, or a for/while without a cleanup region).
      }],
      /*retTy=*/"::cir::CleanupKindAttr",
      /*methodName=*/"maybeGetCleanupKind",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {};"
    >,
    InterfaceMethod<[{
        Returns the first region to be executed in the loop.
      }],
      /*retTy=*/"mlir::Region &",
      /*methodName=*/"getEntry",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return $_op.getCond();"
    >,
    InterfaceMethod<[{
        Returns a list of regions in order of execution.
      }],
      /*retTy=*/"llvm::SmallVector<mlir::Region *>",
      /*methodName=*/"getRegionsInExecutionOrder",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::SmallVector<mlir::Region *, 2>{&$_op.getRegion(0), &$_op.getRegion(1)};
      }]
    >,
    InterfaceMethod<[{
        Recursively walks the body of the loop in pre-order while skipping
        nested loops and executing a callback on every other operation.
      }],
      /*retTy=*/"mlir::WalkResult",
      /*methodName=*/"walkBodySkippingNestedLoops",
      /*args=*/(ins "::llvm::function_ref<mlir::WalkResult (mlir::Operation *)>":$callback),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getBody().template walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
          if (mlir::isa<LoopOpInterface>(op))
            return mlir::WalkResult::skip();
          return callback(op);
        });
      }]
    >
  ];

  let extraClassDeclaration = [{
    /// Generic method to retrieve the successors of a LoopOpInterface operation.
    static void getLoopOpSuccessorRegions(
        ::cir::LoopOpInterface op, ::mlir::RegionBranchPoint point,
        ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
    /// Generic method to retrieve the successor inputs of a LoopOpInterface
    /// operation.
    static ::mlir::ValueRange getLoopOpSuccessorInputs(
        ::cir::LoopOpInterface op, ::mlir::RegionSuccessor successor);
  }];

  let verify = [{
    /// Verify invariants of the LoopOpInterface.
    return detail::verifyLoopOpInterface($_op);
  }];
}

#endif // CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE
