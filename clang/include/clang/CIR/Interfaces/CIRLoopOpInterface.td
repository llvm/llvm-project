//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE
#define CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE

include "aiir/IR/OpBase.td"
include "aiir/Interfaces/ControlFlowInterfaces.td"
include "aiir/Interfaces/LoopLikeInterface.td"

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
      /*retTy=*/"aiir::Region &",
      /*methodName=*/"getCond"
    >,
    InterfaceMethod<[{
        Returns the loop's body region.
      }],
      /*retTy=*/"aiir::Region &",
      /*methodName=*/"getBody"
    >,
    InterfaceMethod<[{
        Returns a pointer to the loop's step region or nullptr.
      }],
      /*retTy=*/"aiir::Region *",
      /*methodName=*/"maybeGetStep",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return nullptr;"
    >,
    InterfaceMethod<[{
        Returns the first region to be executed in the loop.
      }],
      /*retTy=*/"aiir::Region &",
      /*methodName=*/"getEntry",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return $_op.getCond();"
    >,
    InterfaceMethod<[{
        Returns a list of regions in order of execution.
      }],
      /*retTy=*/"llvm::SmallVector<aiir::Region *>",
      /*methodName=*/"getRegionsInExecutionOrder",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::SmallVector<aiir::Region *, 2>{&$_op.getRegion(0), &$_op.getRegion(1)};
      }]
    >,
    InterfaceMethod<[{
        Recursively walks the body of the loop in pre-order while skipping
        nested loops and executing a callback on every other operation.
      }],
      /*retTy=*/"aiir::WalkResult",
      /*methodName=*/"walkBodySkippingNestedLoops",
      /*args=*/(ins "::llvm::function_ref<aiir::WalkResult (aiir::Operation *)>":$callback),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getBody().template walk<aiir::WalkOrder::PreOrder>([&](aiir::Operation *op) {
          if (aiir::isa<LoopOpInterface>(op))
            return aiir::WalkResult::skip();
          return callback(op);
        });
      }]
    >
  ];

  let extraClassDeclaration = [{
    /// Generic method to retrieve the successors of a LoopOpInterface operation.
    static void getLoopOpSuccessorRegions(
        ::cir::LoopOpInterface op, ::aiir::RegionBranchPoint point,
        ::aiir::SmallVectorImpl<::aiir::RegionSuccessor> &regions);
    /// Generic method to retrieve the successor inputs of a LoopOpInterface
    /// operation.
    static ::aiir::ValueRange getLoopOpSuccessorInputs(
        ::cir::LoopOpInterface op, ::aiir::RegionSuccessor successor);
  }];

  let verify = [{
    /// Verify invariants of the LoopOpInterface.
    return detail::verifyLoopOpInterface($_op);
  }];
}

#endif // CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE
