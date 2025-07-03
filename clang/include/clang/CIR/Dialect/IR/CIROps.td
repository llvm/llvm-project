//===-- CIROps.td - CIR dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the CIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIROPS_TD
#define CLANG_CIR_DIALECT_IR_CIROPS_TD

include "clang/CIR/Dialect/IR/CIRDialect.td"
include "clang/CIR/Dialect/IR/CIRTypes.td"
include "clang/CIR/Dialect/IR/CIRAttrs.td"
include "clang/CIR/Dialect/IR/CIRAttrConstraints.td"

include "clang/CIR/Interfaces/CIROpInterfaces.td"
include "clang/CIR/Interfaces/CIRLoopOpInterface.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// CIR Ops
//===----------------------------------------------------------------------===//

// LLVMLoweringInfo is used by cir-tablegen to generate LLVM lowering logic
// automatically for CIR operations. The `llvmOp` field gives the name of the
// LLVM IR dialect operation that the CIR operation will be lowered to. The
// input arguments of the CIR operation will be passed in the same order to the
// lowered LLVM IR operation.
//
// Example:
//
// For the following CIR operation definition:
//
//   def FooOp : CIR_Op<"foo"> {
//     // ...
//     let arguments = (ins CIR_AnyType:$arg1, CIR_AnyType:$arg2);
//     let llvmOp = "BarOp";
//   }
//
// cir-tablegen will generate LLVM lowering code for the FooOp similar to the
// following:
//
//   class CIRFooOpLowering
//       : public mlir::OpConversionPattern<cir::FooOp> {
//   public:
//     using OpConversionPattern<cir::FooOp>::OpConversionPattern;
//
//     mlir::LogicalResult matchAndRewrite(
//         cir::FooOp op,
//         OpAdaptor adaptor,
//         mlir::ConversionPatternRewriter &rewriter) const override {
//       rewriter.replaceOpWithNewOp<mlir::LLVM::BarOp>(
//         op, adaptor.getOperands()[0], adaptor.getOperands()[1]);
//       return mlir::success();
//     }
//   }
//
// If you want fully customized LLVM IR lowering logic, simply exclude the
// `llvmOp` field from your CIR operation definition.
class LLVMLoweringInfo {
  string llvmOp = "";
}

class CIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIR_Dialect, mnemonic, traits>, LLVMLoweringInfo;

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CIR_CastKind : CIR_I32EnumAttr<"CastKind", "cast kind", [
  I32EnumAttrCase<"bitcast", 1>,
  // CK_LValueBitCast
  // CK_LValueToRValueBitCast
  // CK_LValueToRValue
  // CK_NoOp
  // CK_BaseToDerived
  // CK_DerivedToBase
  // CK_UncheckedDerivedToBase
  // CK_Dynamic
  // CK_ToUnion
  I32EnumAttrCase<"array_to_ptrdecay", 11>,
  // CK_FunctionToPointerDecay
  // CK_NullToPointer
  // CK_NullToMemberPointer
  // CK_BaseToDerivedMemberPointer
  // CK_DerivedToBaseMemberPointer
  I32EnumAttrCase<"member_ptr_to_bool", 17>,
  // CK_ReinterpretMemberPointer
  // CK_UserDefinedConversion
  // CK_ConstructorConversion
  I32EnumAttrCase<"int_to_ptr", 21>,
  I32EnumAttrCase<"ptr_to_int", 22>,
  I32EnumAttrCase<"ptr_to_bool", 23>,
  // CK_ToVoid
  // CK_MatrixCast
  // CK_VectorSplat
  I32EnumAttrCase<"integral", 27>,
  I32EnumAttrCase<"int_to_bool", 28>,
  I32EnumAttrCase<"int_to_float", 29>,
  // CK_FloatingToFixedPoint
  // CK_FixedPointToFloating
  // CK_FixedPointCast
  // CK_FixedPointToIntegral
  // CK_IntegralToFixedPoint
  // CK_FixedPointToBoolean
  I32EnumAttrCase<"float_to_int", 36>,
  I32EnumAttrCase<"float_to_bool", 37>,
  I32EnumAttrCase<"bool_to_int", 38>,
  I32EnumAttrCase<"floating", 39>,
  // CK_CPointerToObjCPointerCast
  // CK_BlockPointerToObjCPointerCast
  // CK_AnyPointerToBlockPointerCast
  // CK_ObjCObjectLValueCast
  // I32EnumAttrCase<"float_to_complex", 44>,
  // I32EnumAttrCase<"float_complex_to_real", 45>,
  // I32EnumAttrCase<"float_complex_to_bool", 46>,
  I32EnumAttrCase<"float_complex", 47>,
  // I32EnumAttrCase<"float_complex_to_int_complex", 48>,
  // I32EnumAttrCase<"int_to_complex", 49>,
  I32EnumAttrCase<"int_complex_to_real", 50>,
  I32EnumAttrCase<"int_complex_to_bool", 51>,
  I32EnumAttrCase<"int_complex", 52>,
  I32EnumAttrCase<"int_complex_to_float_complex", 53>,
  // CK_ARCProduceObject
  // CK_ARCConsumeObject
  // CK_ARCReclaimReturnedObject
  // CK_ARCExtendBlockObject
  // CK_AtomicToNonAtomic
  // CK_NonAtomicToAtomic
  // CK_CopyAndAutoreleaseBlockObject
  // CK_BuiltinFnToFnPtr
  // CK_ZeroToOCLOpaqueType
  I32EnumAttrCase<"address_space", 63>,
  // CK_IntToOCLSampler
  // CK_HLSLVectorTruncation
  // CK_HLSLArrayRValue
  // CK_HLSLElementwiseCast
  // CK_HLSLAggregateSplatCast

  // Enums below are specific to CIR and don't have a correspondence to classic
  // codegen:
  I32EnumAttrCase<"bool_to_float", 1000>,
]>;

def CastOp : CIR_Op<"cast",
             [Pure,
              DeclareOpInterfaceMethods<PromotableOpInterface>]> {
  // FIXME: not all conversions are free of side effects.
  let summary = "Conversion between values of different types";
  let description = [{
    Apply the usual C/C++ conversion rules between values. This operation models
    a subset of conversions as defined in Clang's `OperationKinds.def`
    (`llvm-project/clang/include/clang/AST/OperationKinds.def`).

    Note: not all conversions are implemented using `cir.cast`. For instance,
    lvalue-to-rvalue conversion is modeled as a `cir.load` instead.  Currently
    supported kinds:

    - `bitcast`
    - `array_to_ptrdecay`
    - `member_ptr_to_bool
    - `int_to_ptr`
    - `ptr_to_int`
    - `ptr_to_bool`
    - `integral`
    - `int_to_bool`
    - `int_to_float`
    - `float_to_int`
    - `float_to_bool`
    - `bool_to_int`
    - `floating`
    - `float_complex`
    - `int_complex_to_real`
    - `int_complex_to_bool`
    - `int_complex`
    - `int_complex_to_float_complex`
    - `address_space`

    CIR also supports some additional conversions that are not part of the classic
    Clang codegen:

    - `bool_to_float`

    Example:

    ```mlir
    %4 = cir.cast(int_to_bool, %3 : i32), !cir.bool
    ...
    %x = cir.cast(array_to_ptrdecay, %0 : !cir.ptr<!cir.array<i32 x 10>>), !cir.ptr<i32>
    ```
  }];

  let arguments = (ins CIR_CastKind:$kind, CIR_AnyType:$src);
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    `(` $kind `,` $src `:` type($src) `)`
    `,` type($result) attr-dict
  }];

  // The input and output types should match the cast kind.
  let hasVerifier = 1;
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// PtrStrideOp
//===----------------------------------------------------------------------===//

def PtrStrideOp : CIR_Op<"ptr_stride",
                         [Pure, AllTypesMatch<["base", "result"]>]> {
  let summary = "Pointer access with stride";
  let description = [{
    Given a base pointer as first operand, provides a new pointer after applying
    a stride (second operand).

    ```mlir
    %3 = cir.const 0 : i32
    %4 = cir.ptr_stride(%2 : !cir.ptr<i32>, %3 : i32), !cir.ptr<i32>
    ```
  }];

  let arguments = (ins
    CIR_PointerType:$base,
    CIR_AnyFundamentalIntType:$stride
  );

  let results = (outs CIR_PointerType:$result);

  let assemblyFormat = [{
    `(` $base `:` qualified(type($base)) `,` $stride `:`
    qualified(type($stride)) `)` `,` qualified(type($result)) attr-dict
  }];

  let extraClassDeclaration = [{
    // Get type pointed by the base pointer.
    mlir::Type getElementTy() {
      return getBase().getType().getPointee();
    }
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : CIR_Op<"const",
                        [ConstantLike, Pure, AllTypesMatch<["value", "res"]>]> {
  let summary = "Defines a CIR constant";
  let description = [{
    The `cir.const` operation turns a literal into an SSA value. The data is
    attached to the operation as an attribute.

    ```mlir
      %0 = cir.const 42 : i32
      %1 = cir.const 4.2 : f32
      %2 = cir.const nullptr : !cir.ptr<i32>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs CIR_AnyType:$res);

  let assemblyFormat = "attr-dict $value";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isNullPtr() {
      if (const auto ptrAttr = mlir::dyn_cast<cir::ConstPtrAttr>(getValue()))
        return ptrAttr.isNullValue();
      return false;
    }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

class AllocaTypesMatchWith<string summary, string lhsArg, string rhsArg,
                     string transform, string comparator = "std::equal_to<>()">
  : PredOpTrait<summary, CPred<
      comparator # "(" #
      !subst("$_self", "$" # lhsArg # ".getType()", transform) #
             ", $" # rhsArg # ")">> {
  string lhs = lhsArg;
  string rhs = rhsArg;
  string transformer = transform;
}

def AllocaOp : CIR_Op<"alloca", [
  AllocaTypesMatchWith<"'allocaType' matches pointee type of 'addr'",
                 "addr", "allocaType",
                 "cast<PointerType>($_self).getPointee()">,
                 DeclareOpInterfaceMethods<PromotableAllocationOpInterface>]> {
  let summary = "Defines a scope-local variable";
  let description = [{
    The `cir.alloca` operation defines a scope-local variable.

    The presence of the `const` attribute indicates that the local variable is
    declared with C/C++ `const` keyword.

    The result type is a pointer to the input's type.

    Example:

    ```mlir
    // int count;
    %0 = cir.alloca i32, !cir.ptr<i32>, ["count"] {alignment = 4 : i64}

    // int *ptr;
    %1 = cir.alloca !cir.ptr<i32>, !cir.ptr<!cir.ptr<i32>>, ["ptr"] {alignment = 8 : i64}
    ...
    ```
  }];

  let arguments = (ins
    TypeAttr:$allocaType,
    StrAttr:$name,
    UnitAttr:$init,
    UnitAttr:$constant,
    ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment,
    OptionalAttr<ArrayAttr>:$annotations
  );

  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$addr,
                   "mlir::Type":$allocaType,
                   "llvm::StringRef":$name,
                   "mlir::IntegerAttr":$alignment)>
  ];

  let extraClassDeclaration = [{
    // Whether the alloca input type is a pointer.
    bool isPointerType() { return ::mlir::isa<::cir::PointerType>(getAllocaType()); }
  }];

  let assemblyFormat = [{
    $allocaType `,` qualified(type($addr)) `,`
    `[` $name
       (`,` `init` $init^)?
       (`,` `const` $constant^)?
    `]`
    ($annotations^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : CIR_Op<"load", [
  TypesMatchWith<"type of 'result' matches pointee type of 'addr'",
                 "addr", "result",
                 "cast<PointerType>($_self).getPointee()">,
                 DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {

  let summary = "Load value from memory adddress";
  let description = [{
    `cir.load` reads a value (lvalue to rvalue conversion) given an address
    backed up by a `cir.ptr` type. A unit attribute `deref` can be used to
    mark the resulting value as used by another operation to dereference
    a pointer.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = cir.load %0 : !cir.ptr<i32>, i32

    // Load address from memory at address %0. %3 is used by at least one
    // operation that dereferences a pointer.
    %3 = cir.load deref %0 : !cir.ptr<!cir.ptr<i32>>
    ```
  }];

  let arguments = (ins Arg<CIR_PointerType, "the address to load from",
                           [MemRead]>:$addr,
                       UnitAttr:$isDeref,
                       OptionalAttr<I64Attr>:$alignment
                       );
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    (`deref` $isDeref^)?
    (`align` `(` $alignment^ `)`)?
    $addr `:` qualified(type($addr)) `,` type($result) attr-dict
  }];

  // FIXME: add verifier.
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : CIR_Op<"store", [
  TypesMatchWith<"type of 'value' matches pointee type of 'addr'",
                 "addr", "value",
                 "cast<PointerType>($_self).getPointee()">,
                 DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {

  let summary = "Store value to memory address";
  let description = [{
    `cir.store` stores a value (first operand) to the memory address specified
    in the second operand. A unit attribute `volatile` can be used to indicate
    a volatile store. Store's can be marked atomic by using
    `atomic(<mem_order>)`.

    `align` can be used to specify an alignment that's different from the
    default, which is computed from `result`'s type ABI data layout.

    Example:

    ```mlir
    // Store a function argument to local storage, address in %0.
    cir.store %arg0, %0 : i32, !cir.ptr<i32>
    ```
  }];

  let arguments = (ins CIR_AnyType:$value,
                       Arg<CIR_PointerType, "the address to store the value",
                           [MemWrite]>:$addr,
                           OptionalAttr<I64Attr>:$alignment);

  let assemblyFormat = [{
    (`align` `(` $alignment^ `)`)?
    $value `,` $addr attr-dict `:` type($value) `,` qualified(type($addr))
  }];

  // FIXME: add verifier.
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : CIR_Op<"return", [ParentOneOf<["FuncOp", "ScopeOp", "IfOp",
                                              "SwitchOp", "DoWhileOp","WhileOp",
                                              "ForOp", "CaseOp"]>,
                                 Terminator]> {
  let summary = "Return from function";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation.

    ```mlir
      func @foo() -> i32 {
        ...
        cir.return %0 : i32
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<CIR_AnyType>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : CIR_Op<"if",
     [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]>{

  let summary = "the if-then-else operation";
  let description = [{
    The `cir.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand is a `cir.bool`
    type.

    Examples:

    ```mlir
    cir.if %cond  {
      ...
    } else {
      ...
    }

    cir.if %cond  {
      ...
    }

    cir.if %cond  {
      ...
      cir.br ^a
    ^a:
      cir.yield
    }
    ```

    `cir.if` defines no values and the 'else' can be omitted. The if/else
    regions must be terminated. If the region has only one block, the terminator
    can be left out, and `cir.yield` terminator will be inserted implictly.
    Otherwise, the region must be explicitly terminated.
  }];
  let arguments = (ins CIR_BoolType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
  let hasCustomAssemblyFormat=1;
  let skipDefaultBuilders=1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$cond, "bool":$withElseRegion,
      CArg<"BuilderCallbackRef", "buildTerminatedBody">:$thenBuilder,
      CArg<"BuilderCallbackRef", "nullptr">:$elseBuilder)>
  ];
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

def ConditionOp : CIR_Op<"condition", [
  Terminator,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                            ["getSuccessorRegions"]>
]> {
  let summary = "Loop continuation condition.";
  let description = [{
    The `cir.condition` terminates conditional regions. It takes a single
    `cir.bool` operand and, depending on its value, may branch to different
    regions:

     - When in the `cond` region of a loop, it continues the loop
       if true, or exits it if false.
     - When in the `ready` region of a `cir.await`, it branches to the `resume`
       region when true, and to the `suspend` region when false.

    Example:

    ```mlir
    cir.for cond {
      cir.condition(%val) // Branches to `step` region or exits.
    } body {
      cir.yield
    } step {
      cir.yield
    }

    cir.await(user, ready : {
      cir.condition(%arg0) // Branches to `resume` or `suspend` region.
    }, suspend : {
      [...]
    }, resume : {
      [...]
    },)
    ```
  }];
  let arguments = (ins CIR_BoolType:$condition);
  let assemblyFormat = " `(` $condition `)` attr-dict ";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : CIR_Op<"yield", [ReturnLike, Terminator,
                               ParentOneOf<["CaseOp", "DoWhileOp", "ForOp",
                                            "IfOp", "ScopeOp", "SwitchOp",
                                            "TernaryOp", "WhileOp"]>]> {
  let summary = "Represents the default branching behaviour of a region";
  let description = [{
    The `cir.yield` operation terminates regions on different CIR operations,
    and it is used to represent the default branching behaviour of a region.
    Said branching behaviour is determinted by the parent operation. For
    example, a yield in a `switch-case` region implies a fallthrough, while
    a yield in a `cir.if` region implies a branch to the exit block, and so
    on.

    In some cases, it might yield an SSA value and the semantics of how the
    values are yielded is defined by the parent operation. For example, a
    `cir.ternary` operation yields a value from one of its regions.

    As a general rule, `cir.yield` must be explicitly used whenever a region has
    more than one block and no terminator, or within `cir.switch` regions not
    `cir.return` terminated.

    Examples:
    ```mlir
    cir.if %4 {
      ...
      cir.yield
    }

    cir.switch (%5) [
      case (equal, 3) {
        ...
        cir.yield
      }, ...
    ]

    cir.scope {
      ...
      cir.yield
    }

    %x = cir.scope {
      ...
      cir.yield %val
    }

    %y = cir.ternary {
      ...
      cir.yield %val : i32
    } : i32
    ```
  }];

  let arguments = (ins Variadic<CIR_AnyType>:$args);
  let assemblyFormat = "($args^ `:` type($args))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

def BreakOp : CIR_Op<"break", [Terminator]> {
  let summary = "C/C++ `break` statement equivalent";
  let description = [{
    The `cir.break` operation is used to cease the execution of the current loop
    or switch operation and transfer control to the parent operation. It is only
    allowed within a breakable operations (loops and switches).
  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

def ContinueOp : CIR_Op<"continue", [Terminator]> {
  let summary = "C/C++ `continue` statement equivalent";
  let description = [{
    The `cir.continue` operation is used to end execution of the current
    iteration of a loop and resume execution beginning at the next iteration.
    It is only allowed within loop regions.
  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

def ScopeOp : CIR_Op<"scope", [
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "Represents a C/C++ scope";
  let description = [{
    `cir.scope` contains one region and defines a strict "scope" for all new
    values produced within its blocks.

    The region can contain an arbitrary number of blocks but usually defaults
    to one and can optionally return a value (useful for representing values
    coming out of C++ full-expressions) via `cir.yield`:


    ```mlir
    %rvalue = cir.scope {
      ...
      cir.yield %value
    }
    ```

    The blocks can be terminated by `cir.yield`, `cir.return` or `cir.throw`.
    If `cir.scope` yields no value, the `cir.yield` can be left out, and
    will be inserted implicitly.
  }];

  let results = (outs Optional<CIR_AnyType>:$results);
  let regions = (region AnyRegion:$scopeRegion);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    custom<OmittedTerminatorRegion>($scopeRegion) (`:` type($results)^)? attr-dict
  }];

  let extraClassDeclaration = [{
    /// Determine whether the scope is empty, meaning it contains a single block
    /// terminated by a cir.yield.
    bool isEmpty() {
      auto &entry = getRegion().front();
      return getRegion().hasOneBlock() &&
        llvm::isa<YieldOp>(entry.front());
      }
    }];

  let builders = [
    // Scopes for yielding values.
    OpBuilder<(ins
              "llvm::function_ref<void(mlir::OpBuilder &, mlir::Type &, mlir::Location)>":$scopeBuilder)>,
    // Scopes without yielding values.
    OpBuilder<(ins "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$scopeBuilder)>
  ];
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

def CIR_CaseOpKind : CIR_I32EnumAttr<"CaseOpKind", "case kind", [
  I32EnumAttrCase<"Default", 0, "default">,
  I32EnumAttrCase<"Equal", 1, "equal">,
  I32EnumAttrCase<"Anyof", 2, "anyof">,
  I32EnumAttrCase<"Range", 3, "range">
]>;

def CaseOp : CIR_Op<"case", [
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope]> {
  let summary = "Case operation";
  let description = [{
    The `cir.case` operation represents a case within a C/C++ switch.
    The `cir.case` operation must be in a `cir.switch` operation directly
    or indirectly.

    The `cir.case` have 4 kinds:
    - `equal, <constant>`: equality of the second case operand against the
    condition.
    - `anyof, [constant-list]`: equals to any of the values in a subsequent
    following list.
    - `range, [lower-bound, upper-bound]`: the condition is within the closed
                                           interval.
    - `default`: any other value.

    Each case region must be explicitly terminated.
  }];

  let arguments = (ins ArrayAttr:$value, CIR_CaseOpKind:$kind);
  let regions = (region AnyRegion:$caseRegion);

  let assemblyFormat = "`(` $kind `,` $value `)` $caseRegion attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins "mlir::ArrayAttr":$value,
                   "CaseOpKind":$kind,
                   "mlir::OpBuilder::InsertPoint &":$insertPoint)>
  ];
}

def SwitchOp : CIR_Op<"switch",
      [SameVariadicOperandSize,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "Switch operation";
  let description = [{
    The `cir.switch` operation represents C/C++ switch functionality for
    conditionally executing multiple regions of code. The operand to an switch
    is an integral condition value.

    The set of `cir.case` operations and their enclosing `cir.switch`
    represent the semantics of a C/C++ switch statement. Users can use
    `collectCases(llvm::SmallVector<CaseOp> &cases)` to collect the `cir.case`
    operation in the `cir.switch` operation easily.

    The `cir.case` operations don't have to be in the region of `cir.switch`
    directly. However, when all the `cir.case` operations live in the region
    of `cir.switch` directly and there are no other operations except the ending
    `cir.yield` operation in the region of `cir.switch` directly, we say the
    `cir.switch` operation is in a simple form. Users can use
    `bool isSimpleForm(llvm::SmallVector<CaseOp> &cases)` member function to
    detect if the `cir.switch` operation is in a simple form. The simple form
    makes it easier for analyses to handle the `cir.switch` operation
    and makes the boundary to give up clear.

    To make the simple form as common as possible, CIR code generation attaches
    operations corresponding to the statements that lives between top level
    cases into the closest `cir.case` operation.

    For example,

    ```
    switch(int cond) {
      case 4:
        a++;
        b++;
      case 5:
        c++;

      ...
    }
    ```

    The statement `b++` is not a sub-statement of the case statement `case 4`.
    But to make the generated `cir.switch` a simple form, we will attach the
    statement `b++` into the closest `cir.case` operation. So that the generated
    code will be like:

    ```
    cir.switch(int cond) {
      cir.case(equal, 4) {
        a++;
        b++;
        cir.yield
      }
      cir.case(equal, 5) {
        c++;
        cir.yield
      }
      ...
    }
    ```

    For the same reason, we will hoist the case statement as the substatement
    of another case statement so that they will be in the same level. For
    example,

    ```
    switch(int cond) {
      case 4:
      default;
      case 5:
        a++;
      ...
    }
    ```

    will be generated as

    ```
    cir.switch(int cond) {
      cir.case(equal, 4) {
        cir.yield
      }
      cir.case(default) {
        cir.yield
      }
      cir.case(equal, 5) {
        a++;
        cir.yield
      }
      ...
    }
    ```

    The cir.switch is not be considered "simple" if any of the following is
    true:
    - There are case statements of the switch statement that are scope
      other than the top level compound statement scope. Note that a case
      statement itself doesn't form a scope.
    - The sub-statement of the switch statement is not a compound statement.
    - There is any code before the first case statement. For example,

    ```
    switch(int cond) {
      l:
        b++;

      case 4:
        a++;
        break;

      case 5:
        goto l;
      ...
    }
    ```

    the generated CIR for this non-simple switch would be:

    ```
    cir.switch(int cond) {
      cir.label "l"
      b++;
      cir.case(4) {
        a++;
        cir.break
      }
      cir.case(5) {
        goto "l"
      }
      cir.yield
    }
    ```
  }];

  let arguments = (ins CIR_IntType:$condition);

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$condition,
               "BuilderOpStateCallbackRef":$switchBuilder)>
  ];

  let assemblyFormat = [{
    custom<SwitchOp>(
      $body, $condition, type($condition)
    )
    attr-dict
  }];

  let extraClassDeclaration = [{
    // Collect cases in the switch.
    void collectCases(llvm::SmallVectorImpl<CaseOp> &cases);

    // Check if the switch is in a simple form.
    // If yes, collect the cases to \param cases.
    // This is an expensive and need to be used with caution.
    bool isSimpleForm(llvm::SmallVectorImpl<CaseOp> &cases);
  }];
}

//===----------------------------------------------------------------------===//
// SwitchFlatOp
//===----------------------------------------------------------------------===//

def SwitchFlatOp : CIR_Op<"switch.flat", [AttrSizedOperandSegments,
                                          Terminator]> {

  let description = [{
    The `cir.switch.flat` operation is a region-less and simplified
    version of the `cir.switch`.
    Its representation is closer to LLVM IR dialect
    than the C/C++ language feature.
  }];

  let arguments = (ins
    CIR_IntType:$condition,
    Variadic<AnyType>:$defaultOperands,
    VariadicOfVariadic<AnyType, "case_operand_segments">:$caseOperands,
    ArrayAttr:$caseValues,
    DenseI32ArrayAttr:$case_operand_segments
  );

  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );

  let assemblyFormat = [{
    $condition `:` type($condition) `,`
    $defaultDestination (`(` $defaultOperands^ `:` type($defaultOperands) `)`)?
    custom<SwitchFlatOpCases>(ref(type($condition)), $caseValues,
                              $caseDestinations, $caseOperands,
                              type($caseOperands))
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$condition,
      "mlir::Block *":$defaultDestination,
      "mlir::ValueRange":$defaultOperands,
      CArg<"llvm::ArrayRef<llvm::APInt>", "{}">:$caseValues,
      CArg<"mlir::BlockRange", "{}">:$caseDestinations,
      CArg<"llvm::ArrayRef<mlir::ValueRange>", "{}">:$caseOperands)>
  ];
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

def BrOp : CIR_Op<"br",
      [DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     Pure, Terminator]> {
  let summary = "Unconditional branch";
  let description = [{
    The `cir.br` branches unconditionally to a block. Used to represent C/C++
    goto's and general block branching.

    Note that for source level `goto`'s crossing scope boundaries, those are
    usually represented with the "symbolic" `cir.goto` operation.

    Example:

    ```mlir
      ...
        cir.br ^bb3
      ^bb3:
        cir.return
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::Block *":$dest,
              CArg<"mlir::ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>
  ];

  let arguments = (ins Variadic<CIR_AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

def CIR_UnaryOpKind : CIR_I32EnumAttr<"UnaryOpKind", "unary operation kind", [
  I32EnumAttrCase<"Inc",   0, "inc">,
  I32EnumAttrCase<"Dec",   1, "dec">,
  I32EnumAttrCase<"Plus",  2, "plus">,
  I32EnumAttrCase<"Minus", 3, "minus">,
  I32EnumAttrCase<"Not",   4, "not">
]>;

def UnaryOp : CIR_Op<"unary", [Pure, SameOperandsAndResultType]> {
  let summary = "Unary operations";
  let description = [{
    `cir.unary` performs the unary operation according to
    the specified opcode kind: [inc, dec, plus, minus, not].

    It requires one input operand and has one result, both types
    should be the same.

    If the `nsw` (no signed wrap) attribute is present, the result is poison if
    signed overflow occurs.

    ```mlir
    %7 = cir.unary(inc, %1) : i32 -> i32
    %8 = cir.unary(dec, %2) nsw : i32 -> i32
    ```
  }];

  let arguments = (ins
    Arg<CIR_UnaryOpKind, "unary op kind">:$kind,
    Arg<CIR_AnyType>:$input,
    UnitAttr:$no_signed_wrap
  );

  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
      `(` $kind `,` $input `)`
      (`nsw` $no_signed_wrap^)?
      `:` type($input) `,` type($result) attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// BrCondOp
//===----------------------------------------------------------------------===//

def BrCondOp : CIR_Op<"brcond",
      [DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
       Pure, Terminator, AttrSizedOperandSegments]> {
  let summary = "Conditional branch";
  let description = [{
    The `cir.brcond %cond, ^bb0, ^bb1` branches to 'bb0' block in case
    %cond (which must be a !cir.bool type) evaluates to true, otherwise
    it branches to 'bb1'.

    Example:

    ```mlir
      ...
        cir.brcond %a, ^bb3, ^bb4
      ^bb3:
        cir.return
      ^bb4:
        cir.yield
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$cond, "mlir::Block *":$destTrue, "mlir::Block *":$destFalse,
               CArg<"mlir::ValueRange", "{}">:$destOperandsTrue,
               CArg<"mlir::ValueRange", "{}">:$destOperandsFalse), [{
      build($_builder, $_state, cond, destOperandsTrue,
            destOperandsFalse, destTrue, destFalse);
    }]>
  ];

  let arguments = (ins CIR_BoolType:$cond,
                       Variadic<CIR_AnyType>:$destOperandsTrue,
                       Variadic<CIR_AnyType>:$destOperandsFalse);
  let successors = (successor AnySuccessor:$destTrue, AnySuccessor:$destFalse);
  let assemblyFormat = [{
    $cond
    $destTrue (`(` $destOperandsTrue^ `:` type($destOperandsTrue) `)`)?
    `,`
    $destFalse (`(` $destOperandsFalse^ `:` type($destOperandsFalse) `)`)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Common loop op definitions
//===----------------------------------------------------------------------===//

class LoopOpBase<string mnemonic> : CIR_Op<mnemonic, [
  LoopOpInterface,
  NoRegionArguments,
]> {
  let extraClassDefinition = [{
    void $cppClass::getSuccessorRegions(
        mlir::RegionBranchPoint point,
        llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
      LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
    }
    llvm::SmallVector<Region *> $cppClass::getLoopRegions() {
      return {&getBody()};
    }
  }];
}

//===----------------------------------------------------------------------===//
// While & DoWhileOp
//===----------------------------------------------------------------------===//

class WhileOpBase<string mnemonic> : LoopOpBase<mnemonic> {
  defvar isWhile = !eq(mnemonic, "while");
  let summary = "C/C++ " # !if(isWhile, "while", "do-while") # " loop";
  let builders = [
    OpBuilder<(ins "BuilderCallbackRef":$condBuilder,
                   "BuilderCallbackRef":$bodyBuilder), [{
        mlir::OpBuilder::InsertionGuard guard($_builder);
        $_builder.createBlock($_state.addRegion());
      }] # !if(isWhile, [{
        condBuilder($_builder, $_state.location);
        $_builder.createBlock($_state.addRegion());
        bodyBuilder($_builder, $_state.location);
      }], [{
        bodyBuilder($_builder, $_state.location);
        $_builder.createBlock($_state.addRegion());
        condBuilder($_builder, $_state.location);
      }])>
  ];
}

def WhileOp : WhileOpBase<"while"> {
  let regions = (region SizedRegion<1>:$cond, MinSizedRegion<1>:$body);
  let assemblyFormat = "$cond `do` $body attr-dict";

  let description = [{
    Represents a C/C++ while loop. It consists of two regions:

     - `cond`: single block region with the loop's condition. Should be
     terminated with a `cir.condition` operation.
     - `body`: contains the loop body and an arbitrary number of blocks.

    Example:

    ```mlir
    cir.while {
      cir.break
    ^bb2:
      cir.yield
    } do {
      cir.condition %cond : cir.bool
    }
    ```
  }];
}

def DoWhileOp : WhileOpBase<"do"> {
  let regions = (region MinSizedRegion<1>:$body, SizedRegion<1>:$cond);
  let assemblyFormat = " $body `while` $cond attr-dict";

  let extraClassDeclaration = [{
    mlir::Region &getEntry() { return getBody(); }
  }];

  let description = [{
    Represents a C/C++ do-while loop. Identical to `cir.while` but the
    condition is evaluated after the body.

    Example:

    ```mlir
    cir.do {
      cir.break
    ^bb2:
      cir.yield
    } while {
      cir.condition %cond : cir.bool
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

def ForOp : LoopOpBase<"for"> {
  let summary = "C/C++ for loop counterpart";
  let description = [{
    Represents a C/C++ for loop. It consists of three regions:

     - `cond`: single block region with the loop's condition. Should be
     terminated with a `cir.condition` operation.
     - `body`: contains the loop body and an arbitrary number of blocks.
     - `step`: single block region with the loop's step.

    Example:

    ```mlir
    cir.for cond {
      cir.condition(%val)
    } body {
      cir.break
    ^bb2:
      cir.yield
    } step {
      cir.yield
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$cond,
                        MinSizedRegion<1>:$body,
                        SizedRegion<1>:$step);
  let assemblyFormat = [{
    `:` `cond` $cond
    `body` $body
    `step` $step
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$condBuilder,
                   "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$bodyBuilder,
                   "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$stepBuilder), [{
        mlir::OpBuilder::InsertionGuard guard($_builder);

        // Build condition region.
        $_builder.createBlock($_state.addRegion());
        condBuilder($_builder, $_state.location);

        // Build body region.
        $_builder.createBlock($_state.addRegion());
        bodyBuilder($_builder, $_state.location);

        // Build step region.
        $_builder.createBlock($_state.addRegion());
        stepBuilder($_builder, $_state.location);
      }]>
  ];

  let extraClassDeclaration = [{
    mlir::Region *maybeGetStep() { return &getStep(); }
    llvm::SmallVector<mlir::Region *> getRegionsInExecutionOrder() {
      return llvm::SmallVector<mlir::Region *, 3>{&getCond(), &getBody(), &getStep()};
    }
  }];
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def CIR_CmpOpKind : CIR_I32EnumAttr<"CmpOpKind", "compare operation kind", [
  I32EnumAttrCase<"lt", 0>,
  I32EnumAttrCase<"le", 1>,
  I32EnumAttrCase<"gt", 2>,
  I32EnumAttrCase<"ge", 3>,
  I32EnumAttrCase<"eq", 4>,
  I32EnumAttrCase<"ne", 5>
]>;

def CmpOp : CIR_Op<"cmp", [Pure, SameTypeOperands]> {

  let summary = "Compare values two values and produce a boolean result";
  let description = [{
    `cir.cmp` compares two input operands of the same type and produces a
    `cir.bool` result. The kinds of comparison available are:
    [lt,gt,ge,eq,ne]

    ```mlir
    %7 = cir.cmp(gt, %1, %2) : i32, !cir.bool
    ```
  }];

  let arguments = (ins
    CIR_CmpOpKind:$kind,
    CIR_AnyType:$lhs,
    CIR_AnyType:$rhs
  );

  let results = (outs CIR_BoolType:$result);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) `,` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BinOp
//===----------------------------------------------------------------------===//

// FIXME: represent Commutative, Idempotent traits for appropriate binops
def CIR_BinOpKind : CIR_I32EnumAttr<
  "BinOpKind", "binary operation (arith and logic) kind", [
    I32EnumAttrCase<"Mul", 0, "mul">,
    I32EnumAttrCase<"Div", 1, "div">,
    I32EnumAttrCase<"Rem", 2, "rem">,
    I32EnumAttrCase<"Add", 3, "add">,
    I32EnumAttrCase<"Sub", 4, "sub">,
    I32EnumAttrCase<"And", 5, "and">,
    I32EnumAttrCase<"Xor", 6, "xor">,
    I32EnumAttrCase<"Or", 7, "or">,
    I32EnumAttrCase<"Max", 8, "max">
]>;

def BinOp : CIR_Op<"binop", [Pure,
  SameTypeOperands, SameOperandsAndResultType]> {

  let summary = "Binary operations (arith and logic)";
  let description = [{
    cir.binop performs the binary operation according to
    the specified opcode kind: [mul, div, rem, add, sub,
    and, xor, or, max].

    It requires two input operands and has one result, all types
    should be the same.

    If the `nsw` (no signed wrap) or `nuw` (no unsigned wrap) attributes are
    present, the result is poison if signed or unsigned overflow occurs
    (respectively).

    If the `sat` (saturated) attribute is present, the result is clamped to
    the maximum value representatable by the type if it would otherwise
    exceed that value and is clamped to the minimum representable value if
    it would otherwise be below that value.

    ```mlir
    %5 = cir.binop(add, %1, %2) : !s32i
    %6 = cir.binop(mul, %1, %2) : !u8i
    %7 = cir.binop(add, %1, %2) nsw : !s32i
    %8 = cir.binop(add, %3, %4) nuw : !u32i
    %9 = cir.binop(add, %1, %2) sat : !s32i
    ```
  }];

  let arguments = (ins
    CIR_BinOpKind:$kind,
    CIR_AnyType:$lhs, CIR_AnyType:$rhs,
    UnitAttr:$no_unsigned_wrap,
    UnitAttr:$no_signed_wrap,
    UnitAttr:$saturated
  );

  // TODO: get more accurate than CIR_AnyType
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)`
    (`nsw` $no_signed_wrap^)?
    (`nuw` $no_unsigned_wrap^)?
    (`sat` $saturated^)?
    `:` type($lhs) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

def ShiftOp : CIR_Op<"shift", [Pure]> {
  let summary = "Shift";
  let description = [{
    The `cir.shift` operation performs a bitwise shift, either to the left or to
    the right, based on the first operand. The second operand specifies the
    value to be shifted, and the third operand determines the number of
    positions by which the shift is applied, They must be either all vector of
    integer type, or all integer type. If they are vectors, each vector element of
    the shift target is shifted by the corresponding shift amount in
    the shift amount vector.

    ```mlir
    %res = cir.shift(left, %lhs : !u64i, %amount : !s32i) -> !u64i
    %new_vec = cir.shift(left, %lhs : !cir.vector<2 x !s32i>, %rhs :
        !cir.vector<2 x !s32i>) -> !cir.vector<2 x !s32i>
    ```
  }];

  let arguments = (ins
    CIR_AnyIntOrVecOfIntType:$value,
    CIR_AnyIntOrVecOfIntType:$amount,
    UnitAttr:$isShiftleft
  );

  let results = (outs CIR_AnyIntOrVecOfIntType:$result);

  let assemblyFormat = [{
    `(`
      (`left` $isShiftleft^) : (```right`)?
      `,` $value `:` type($value)
      `,` $amount `:` type($amount)
    `)` `->` type($result) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def SelectOp : CIR_Op<"select", [Pure,
    AllTypesMatch<["true_value", "false_value", "result"]>]> {
  let summary = "Yield one of two values based on a boolean value";
  let description = [{
    The `cir.select` operation takes three operands. The first operand
    `condition` is a boolean value of type `!cir.bool`. The second and the third
    operand can be of any CIR types, but their types must be the same. If the
    first operand is `true`, the operation yields its second operand. Otherwise,
    the operation yields its third operand.

    Example:

    ```mlir
    %0 = cir.const #cir.bool<true> : !cir.bool
    %1 = cir.const #cir.int<42> : !s32i
    %2 = cir.const #cir.int<72> : !s32i
    %3 = cir.select if %0 then %1 else %2 : (!cir.bool, !s32i, !s32i) -> !s32i
    ```
  }];

  let arguments = (ins CIR_BoolType:$condition, CIR_AnyType:$true_value,
                       CIR_AnyType:$false_value);
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    `if` $condition `then` $true_value `else` $false_value
    `:` `(`
      qualified(type($condition)) `,`
      qualified(type($true_value)) `,`
      qualified(type($false_value))
    `)` `->` qualified(type($result)) attr-dict
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

def TernaryOp : CIR_Op<"ternary",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "The `cond ? a : b` C/C++ ternary operation";
  let description = [{
    The `cir.ternary` operation represents C/C++ ternary, much like a `select`
    operation. The first argument is a `cir.bool` condition to evaluate, followed
    by two regions to execute (true or false). This is different from `cir.if`
    since each region is one block sized and the `cir.yield` closing the block
    scope should have one argument.

    `cir.ternary` also represents the GNU binary conditional operator ?: which
    reuses the parent operation for both the condition and the true branch to
    evaluate it only once.

    Example:

    ```mlir
    // cond = a && b;

    %x = cir.ternary (%cond, true_region {
      ...
      cir.yield %a : i32
    }, false_region {
      ...
      cir.yield %b : i32
    }) -> i32
    ```
  }];
  let arguments = (ins CIR_BoolType:$cond);
  let regions = (region AnyRegion:$trueRegion,
                        AnyRegion:$falseRegion);
  let results = (outs Optional<CIR_AnyType>:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$cond,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$trueBuilder,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$falseBuilder)
      >
  ];

  let assemblyFormat = [{
    `(` $cond `,`
      `true` $trueRegion `,`
      `false` $falseRegion
    `)` `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

// Linkage types. This is currently a replay of llvm/IR/GlobalValue.h, this is
// currently handy as part of forwarding appropriate linkage types for LLVM
// lowering, specially useful for C++ support.

/// An enumeration for the kinds of linkage for global values.
def CIR_GlobalLinkageKind : CIR_I32EnumAttr<
  "GlobalLinkageKind", "linkage kind", [
    // Externally visible function
    I32EnumAttrCase<"ExternalLinkage", 0, "external">,
    // Available for inspection, not emission.
    I32EnumAttrCase<"AvailableExternallyLinkage", 1, "available_externally">,
    // Keep one copy of function when linking (inline)
    I32EnumAttrCase<"LinkOnceAnyLinkage", 2, "linkonce">,
    // Same, but only replaced by something equivalent.
    I32EnumAttrCase<"LinkOnceODRLinkage", 3, "linkonce_odr">,
    // Keep one copy of named function when linking (weak)
    I32EnumAttrCase<"WeakAnyLinkage", 4, "weak">,
    // Same, but only replaced by something equivalent.
    I32EnumAttrCase<"WeakODRLinkage", 5, "weak_odr">,
    // TODO: should we add something like appending linkage too?
    // Special purpose, only applies to global arrays
    // I32EnumAttrCase<"AppendingLinkage", 6, "appending">,
    // Rename collisions when linking (static functions).
    I32EnumAttrCase<"InternalLinkage", 7, "internal">,
    // Like Internal, but omit from symbol table, prefix it with
    // "cir_" to prevent clash with MLIR's symbol "private".
    I32EnumAttrCase<"PrivateLinkage", 8, "cir_private">,
    // ExternalWeak linkage description.
    I32EnumAttrCase<"ExternalWeakLinkage", 9, "extern_weak">,
    // Tentative definitions.
    I32EnumAttrCase<"CommonLinkage", 10, "common">
]>;

// TODO(CIR): For starters, cir.global has only name and type.  The other
// properties of a global variable will be added over time as more of ClangIR
// is upstreamed.

def GlobalOp : CIR_Op<"global",
                      [DeclareOpInterfaceMethods<CIRGlobalValueInterface>]> {
  let summary = "Declare or define a global variable";
  let description = [{
    The `cir.global` operation declares or defines a named global variable.

    The backing memory for the variable is allocated statically and is
    described by the type of the variable.

    The `linkage` tracks C/C++ linkage types, currently very similar to LLVM's.
    Symbol visibility in `sym_visibility` is defined in terms of MLIR's visibility
    and verified to be in accordance to `linkage`.
  }];

  // Note that both sym_name and sym_visibility are tied to Symbol trait.
  // TODO: sym_visibility can possibly be represented by implementing the
  // necessary Symbol's interface in terms of linkage instead.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       DefaultValuedAttr<
                        CIR_VisibilityAttr,
                        "VisibilityKind::Default"
                       >:$global_visibility,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       TypeAttr:$sym_type,
                       CIR_GlobalLinkageKind:$linkage,
                       OptionalAttr<AnyAttr>:$initial_value,
                       UnitAttr:$comdat,
                       UnitAttr:$dso_local,
                       OptionalAttr<I64Attr>:$alignment);

  let assemblyFormat = [{
    ($sym_visibility^)?
    (`` $global_visibility^)?
    $linkage
    (`comdat` $comdat^)?
    (`dso_local` $dso_local^)?
    $sym_name
    custom<GlobalOpTypeAndInitialValue>($sym_type, $initial_value)
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool isDeclaration() { return !getInitialValue(); }
    bool hasInitializer() { return !isDeclaration(); }
  }];

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins
    "llvm::StringRef":$sym_name,
    "mlir::Type":$sym_type,
    // CIR defaults to external linkage.
    CArg<"cir::GlobalLinkageKind",
    "cir::GlobalLinkageKind::ExternalLinkage">:$linkage)>];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

def GetGlobalOp : CIR_Op<"get_global",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get the address of a global variable";
  let description = [{
    The `cir.get_global` operation retrieves the address pointing to a
    named global variable. If the global variable is marked constant, writing
    to the resulting address (such as through a `cir.store` operation) is
    undefined. The resulting type must always be a `!cir.ptr<...>` type with the
    same address space as the global variable.

    Example:
    ```mlir
    %x = cir.get_global @gv : !cir.ptr<i32>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs Res<CIR_PointerType, "", []>:$addr);

  let assemblyFormat = [{
    $name `:` qualified(type($addr)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// GetBitfieldOp
//===----------------------------------------------------------------------===//

def GetBitfieldOp : CIR_Op<"get_bitfield"> {
  let summary = "Get the information for a bitfield member";
  let description = [{
    The `cir.get_bitfield` operation provides a load-like access to
    a bit field of a record.

    It expects a name if a bit field, a pointer to a storage in the
    base record, a type of the storage, a name of the bitfield,
    a size the bit field, an offset of the bit field and a sign.

    A unit attribute `volatile` can be used to indicate a volatile load of the
    bitfield.

    Example:
    Suppose we have a struct with multiple bitfields stored in
    different members. The `cir.get_bitfield` operation gets the value
    of the bitfield.
    ```C++
    typedef struct {
      int a : 4;
      int b : 27;
      int c : 17;
      int d : 2;
      int e : 15;
    } S;

    int load_bitfield(S& s) {
      return s.e;
    }
    ```

    ```mlir
    // 'e' is in the storage with the index 1
    !cir.record<struct "S" packed padded {!u64i, !u16i, !cir.array<!u8i x 2>}>
    #bfi_e = #cir.bitfield_info<name = "e", storage_type = !u16i, size = 15,
                                offset = 0, is_signed = true>

    %2 = cir.load %0 : !cir.ptr<!cir.ptr<!record_type>>, !cir.ptr<!record_type>
    %3 = cir.get_member %2[1] {name = "e"} : !cir.ptr<!record_type>
                                                             -> !cir.ptr<!u16i>
    %4 = cir.get_bitfield(#bfi_e, %3 : !cir.ptr<!u16i>) -> !s32i
    ```
    }];

  let arguments = (ins
    Arg<CIR_PointerType, "the address to load from", [MemRead]>:$addr,
    BitfieldInfoAttr:$bitfield_info,
    UnitAttr:$is_volatile
    );

  let results = (outs CIR_IntType:$result);

  let assemblyFormat = [{ `(`$bitfield_info `,` $addr attr-dict `:`
   qualified(type($addr)) `)` `->` type($result) }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$type,
                   "mlir::Value":$addr,
                   "mlir::Type":$storage_type,
                   "llvm::StringRef":$name,
                   "unsigned":$size,
                   "unsigned":$offset,
                   "bool":$is_signed,
                   "bool":$is_volatile
                   ),
   [{
      BitfieldInfoAttr info =
        BitfieldInfoAttr::get($_builder.getContext(),
                              name, storage_type,
                              size, offset, is_signed);
      build($_builder, $_state, type, addr, info, is_volatile);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// GetMemberOp
//===----------------------------------------------------------------------===//

def GetMemberOp : CIR_Op<"get_member"> {
  let summary = "Get the address of a member of a record";
  let description = [{
    The `cir.get_member` operation gets the address of a particular named
    member from the input record.

    It expects a pointer to the base record as well as the name of the member
    and its field index.

    Example:
    ```mlir
    // Suppose we have a record with multiple members.
    !s32i = !cir.int<s, 32>
    !s8i = !cir.int<s, 8>
    !ty_B = !cir.record<"struct.B" {!s32i, !s8i}>

    // Get the address of the member at index 1.
    %1 = cir.get_member %0[1] {name = "i"} : (!cir.ptr<!ty_B>) -> !cir.ptr<!s8i>
    ```
  }];

  let arguments = (ins
    Arg<CIR_PointerType, "the address to load from", [MemRead]>:$addr,
    StrAttr:$name,
    IndexAttr:$index_attr);

  let results = (outs Res<CIR_PointerType, "">:$result);

  let assemblyFormat = [{
    $addr `[` $index_attr `]` attr-dict
    `:` qualified(type($addr)) `->` qualified(type($result))
  }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$type,
                   "mlir::Value":$value,
                   "llvm::StringRef":$name,
                   "unsigned":$index),
    [{
      mlir::APInt fieldIdx(64, index);
      build($_builder, $_state, type, value, name, fieldIdx);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the index of the record member being accessed.
    uint64_t getIndex() { return getIndexAttr().getZExtValue(); }

    /// Return the record type pointed by the base pointer.
    cir::PointerType getAddrTy() { return getAddr().getType(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// TODO(CIR): FuncOp is still a tiny shell of what it will become.  Many more
// properties and attributes will be added as upstreaming continues.

def FuncOp : CIR_Op<"func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  DeclareOpInterfaceMethods<CIRGlobalValueInterface>,
  IsolatedFromAbove
]> {
  let summary = "Declare or define a function";
  let description = [{
    The `cir.func` operation defines a function, similar to the `mlir::FuncOp`
    built-in.

    The function linkage information is specified by `linkage`, as defined by
    `GlobalLinkageKind` attribute.

    Example:

    ```mlir
    // External function definitions.
    cir.func @abort()

    // A function with internal linkage.
    cir.func internal @count(%x: i64) -> (i64)
      return %x : i64

    // Linkage information
    cir.func linkonce_odr @some_method(...)
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       CIR_VisibilityAttr:$global_visibility,
                       TypeAttrOf<CIR_FuncType>:$function_type,
                       UnitAttr:$dso_local,
                       DefaultValuedAttr<CIR_GlobalLinkageKind,
                                         "cir::GlobalLinkageKind::ExternalLinkage">:$linkage,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       UnitAttr:$comdat,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<FlatSymbolRefAttr>:$aliasee);

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins
    "llvm::StringRef":$sym_name, "FuncType":$type,
    CArg<"cir::GlobalLinkageKind", "cir::GlobalLinkageKind::ExternalLinkage">:$linkage)
  >];

  let extraClassDeclaration = [{
    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion();

    /// Returns the results types that the callable region produces when
    /// executed.
    llvm::ArrayRef<mlir::Type> getCallableResults() {
      return getFunctionType().getReturnTypes();
    }

    /// Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
       return getFunctionType().getInputs();
    }

    /// Returns 0 or 1 result type of this function (0 in the case of a function
    /// returing void)
    llvm::ArrayRef<mlir::Type> getResultTypes() {
       return getFunctionType().getReturnTypes();
    }

    // TODO(cir): this should be an operand attribute, but for now we just hard-
    // wire this as a function. Will later add a $no_proto argument to this op.
    bool getNoProto() {
      assert(!cir::MissingFeatures::opFuncNoProto());
      return false;
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CIR_SideEffect : CIR_I32EnumAttr<
    "SideEffect", "allowed side effects of a function", [
      I32EnumAttrCase<"All", 0, "all">,
      I32EnumAttrCase<"Pure", 1, "pure">,
      I32EnumAttrCase<"Const", 2, "const">
]> {
  let description = [{
    The side effect attribute specifies the possible side effects of the callee
    of a call operation. This is an enumeration attribute and all possible
    enumerators are:

    - all: The callee can have any side effects. This is the default if no side
      effects are explicitly listed.
    - pure: The callee may read data from memory, but it cannot write data to
      memory. This has the same effect as the GNU C/C++ attribute
      `__attribute__((pure))`.
    - const: The callee may not read or write data from memory. This has the
      same effect as the GNU C/C++ attribute `__attribute__((const))`.

    Examples:

    ```mlir
    %2 = cir.call @add(%0, %1) : (!s32i, !s32i) -> !s32i
    %2 = cir.call @add(%0, %1) : (!s32i, !s32i) -> !s32i side_effect(pure)
    %2 = cir.call @add(%0, %1) : (!s32i, !s32i) -> !s32i side_effect(const)
    ```
  }];
}

class CIR_CallOpBase<string mnemonic, list<Trait> extra_traits = []>
    : Op<CIR_Dialect, mnemonic,
         !listconcat(extra_traits,
                     [DeclareOpInterfaceMethods<CIRCallOpInterface>,
                      DeclareOpInterfaceMethods<SymbolUserOpInterface>])> {
  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    mlir::OperandRange getArgOperands();
    mlir::MutableOperandRange getArgOperandsMutable();

    /// Return the callee of this operation
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(),
                       mlir::cast<mlir::SymbolRefAttr>(callee));
    }

    mlir::ArrayAttr getArgAttrsAttr() { return {}; }
    ::mlir::ArrayAttr getResAttrsAttr() { return {}; }

    void setResAttrsAttr(::mlir::ArrayAttr attrs) {}
    void setArgAttrsAttr(::mlir::ArrayAttr attrs) {}

    ::mlir::Attribute removeArgAttrsAttr() { return {}; }
    ::mlir::Attribute removeResAttrsAttr() { return {}; }

    bool isIndirect() { return !getCallee(); }
    mlir::Value getIndirectCall();

    void setArg(unsigned index, mlir::Value value) {
      if (!isIndirect()) {
        setOperand(index, value);
        return;
      }

      // For indirect call, the operand list is shifted by one.
      setOperand(index + 1, value);
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  // TODO(cir): for now cir.call is just a tiny shell of what it will become.
  // More attributes, arguments, and properties will be added in the future as
  // the upstreaming process moves on. The verifiers is also missing for now,
  // will add in the future.

  dag commonArgs = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
      Variadic<CIR_AnyType>:$args,
      UnitAttr:$nothrow,
      DefaultValuedAttr<CIR_SideEffect, "SideEffect::All">:$side_effect);
}

def CallOp : CIR_CallOpBase<"call", [NoRegionArguments]> {
  let summary = "call a function";
  let description = [{
    The `cir.call` operation represents a function call. It could represent
    either a direct call or an indirect call.

    If the operation represents a direct call, the callee should be defined
    within the same symbol scope as the call. The `callee` attribute contains a
    symbol reference to the callee function. All operands of this operation are
    arguments to the callee function.

    If the operation represents an indirect call, the `callee` attribute is
    empty. The first operand of this operation must be a pointer to the callee
    function. The rest operands are arguments to the callee function.

    Example:

    ```mlir
    %0 = cir.call @foo()
    ```
  }];

  let results = (outs Optional<CIR_AnyType>:$result);
  let arguments = commonArgs;

  let builders = [
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee, "mlir::Type":$resType,
                   "mlir::ValueRange":$operands), [{
      $_state.addOperands(operands);
      if (callee)
        $_state.addAttribute("callee", callee);
      if (resType && !isa<VoidType>(resType))
        $_state.addTypes(resType);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// StackSaveOp & StackRestoreOp
//===----------------------------------------------------------------------===//

def StackSaveOp : CIR_Op<"stacksave"> {
  let summary = "remembers the current state of the function stack";
  let description = [{
    Saves current state of the function stack. Returns a pointer to an opaque object
    that later can be passed into cir.stackrestore.
    This is used during the lowering of variable length array allocas.

    This operation corresponds to LLVM intrinsic `stacksave`.

    ```mlir
    %0 = cir.stacksave : <!u8i>
    ```
  }];

  let results = (outs CIR_PointerType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def StackRestoreOp : CIR_Op<"stackrestore"> {
  let summary = "restores the state of the function stack";
  let description = [{
    Restore the state of the function stack to the state it was
    in when the corresponding cir.stacksave executed.
    This is used during the lowering of variable length array allocas.

    This operation corresponds to LLVM intrinsic `stackrestore`.

    ```mlir
    %0 = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"] {alignment = 8 : i64}
    %1 = cir.stacksave : <!u8i>
    cir.store %1, %0 : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
    %2 = cir.load %0 : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
    cir.stackrestore %2 : !cir.ptr<!u8i>
    ```
  }];

  let arguments = (ins CIR_PointerType:$ptr);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr))";
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

def UnreachableOp : CIR_Op<"unreachable", [Terminator]> {
  let summary = "invoke immediate undefined behavior";
  let description = [{
    If the program control flow reaches a `cir.unreachable` operation, the
    program exhibits undefined behavior immediately. This operation is useful
    in cases where the unreachability of a program point needs to be explicitly
    marked.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// TrapOp
//===----------------------------------------------------------------------===//

def TrapOp : CIR_Op<"trap", [Terminator]> {
  let summary = "Exit the program abnormally";
  let description = [{
    The cir.trap operation causes the program to exit abnormally. The
    implementations may implement this operation with different mechanisms. For
    example, an implementation may implement this operation by calling abort,
    while another implementation may implement this operation by executing an
    illegal instruction.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// VecCreate
//===----------------------------------------------------------------------===//

def VecCreateOp : CIR_Op<"vec.create", [Pure]> {

  let summary = "Create a vector value";
  let description = [{
    The `cir.vec.create` operation creates a vector value with the given element
    values. The number of element arguments must match the number of elements
    in the vector type.
  }];

  let arguments = (ins Variadic<CIR_VectorElementType>:$elements);
  let results = (outs CIR_VectorType:$result);

  let assemblyFormat = [{
    `(` ($elements^ `:` type($elements))? `)` `:` qualified(type($result))
    attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VecInsertOp
//===----------------------------------------------------------------------===//

def VecInsertOp : CIR_Op<"vec.insert", [Pure,
  TypesMatchWith<"argument type matches vector element type", "vec", "value",
                 "cast<VectorType>($_self).getElementType()">,
  AllTypesMatch<["result", "vec"]>]> {

  let summary = "Insert one element into a vector object";
  let description = [{
    The `cir.vec.insert` operation produces a new vector by replacing
    the element of the input vector at `index` with `value`.

    ```mlir
    %value = cir.const #cir.int<5> : !s32i
    %index = cir.const #cir.int<2> : !s32i
    %vec_tmp = cir.load %0 : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
    %new_vec = cir.vec.insert %value, %vec_tmp[%index : !s32i] : !cir.vector<4 x !s32i>
    ```
  }];

  let arguments = (ins
    CIR_VectorType:$vec,
    CIR_VectorElementType:$value,
    CIR_AnyFundamentalIntType:$index
  );

  let results = (outs CIR_VectorType:$result);

  let assemblyFormat = [{
    $value `,` $vec `[` $index `:` type($index) `]` attr-dict `:`
    qualified(type($vec))
  }];
}

//===----------------------------------------------------------------------===//
// VecExtractOp
//===----------------------------------------------------------------------===//

def VecExtractOp : CIR_Op<"vec.extract", [Pure,
  TypesMatchWith<"type of 'result' matches element type of 'vec'", "vec",
                 "result", "cast<VectorType>($_self).getElementType()">]> {

  let summary = "Extract one element from a vector object";
  let description = [{
    The `cir.vec.extract` operation extracts the element at the given index
    from a vector object.

    ```mlir
    %tmp = cir.load %vec : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
    %idx = cir.const #cir.int<1> : !s32i
    %element = cir.vec.extract %tmp[%idx : !s32i] : !cir.vector<4 x !s32i>
    ```
  }];

  let arguments = (ins CIR_VectorType:$vec, CIR_AnyFundamentalIntType:$index);
  let results = (outs CIR_VectorElementType:$result);

  let assemblyFormat = [{
    $vec `[` $index `:` type($index) `]` attr-dict `:` qualified(type($vec))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VecCmpOp
//===----------------------------------------------------------------------===//

def VecCmpOp : CIR_Op<"vec.cmp", [Pure, SameTypeOperands]> {

  let summary = "Compare two vectors";
  let description = [{
    The `cir.vec.cmp` operation does an element-wise comparison of two vectors
    of the same type. The result is a vector of the same size as the operands
    whose element type is the signed integral type that is the same size as the
    element type of the operands. The values in the result are 0 or -1.

    ```mlir
    %eq = cir.vec.cmp(eq, %vec_a, %vec_b) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
    %lt = cir.vec.cmp(lt, %vec_a, %vec_b) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
    ```
  }];

  let arguments = (ins
    CIR_CmpOpKind:$kind,
    CIR_VectorType:$lhs,
    CIR_VectorType:$rhs
  );

  let results = (outs CIR_VectorType:$result);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs `)` `:` qualified(type($lhs)) `,`
    qualified(type($result)) attr-dict
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VecShuffleOp
//===----------------------------------------------------------------------===//

// TODO: Create an interface that both VecShuffleOp and VecShuffleDynamicOp
// implement.  This could be useful for passes that don't care how the vector
// shuffle was specified.

def VecShuffleOp : CIR_Op<"vec.shuffle",
                   [Pure, AllTypesMatch<["vec1", "vec2"]>]> {
  let summary = "Combine two vectors using indices passed as constant integers";
  let description = [{
    The `cir.vec.shuffle` operation implements the documented form of Clang's
    `__builtin_shufflevector`, where the indices of the shuffled result are
    integer constants.

    The two input vectors, which must have the same type, are concatenated.
    Each of the integer constant arguments is interpreted as an index into that
    concatenated vector, with a value of -1 meaning that the result value
    doesn't matter. The result vector, which must have the same element type as
    the input vectors and the same number of elements as the list of integer
    constant indices, is constructed by taking the elements at the given
    indices from the concatenated vector. The size of the result vector does
    not have to match the size of the individual input vectors or of the
    concatenated vector.

    ```mlir
    %new_vec = cir.vec.shuffle(%vec_1, %vec_2 : !cir.vector<2 x !s32i>)
        [#cir.int<3> : !s64i, #cir.int<1> : !s64i] : !cir.vector<2 x !s32i>
    ```
  }];

  let arguments = (ins
    CIR_VectorType:$vec1,
    CIR_VectorType:$vec2,
    CIR_IntArrayAttr:$indices
  );

  let results = (outs CIR_VectorType:$result);
  let assemblyFormat = [{
    `(` $vec1 `,` $vec2 `:` qualified(type($vec1)) `)` $indices `:`
     qualified(type($result)) attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VecShuffleDynamicOp
//===----------------------------------------------------------------------===//

def VecShuffleDynamicOp : CIR_Op<"vec.shuffle.dynamic",
                          [Pure, AllTypesMatch<["vec", "result"]>]> {
  let summary = "Shuffle a vector using indices in another vector";
  let description = [{
    The `cir.vec.shuffle.dynamic` operation implements the undocumented form of
    Clang's __builtin_shufflevector, where the indices of the shuffled result
    can be runtime values.

    There are two input vectors, which must have the same number of elements.
    The second input vector must have an integral element type. The elements of
    the second vector are interpreted as indices into the first vector. The
    result vector is constructed by taking the elements from the first input
    vector from the indices indicated by the elements of the second vector.

    ```mlir
    %new_vec = cir.vec.shuffle.dynamic %vec : !cir.vector<4 x !s32i>, %indices
        : !cir.vector<4 x !s32i>
    ```
  }];

  let arguments = (ins CIR_VectorType:$vec, CIR_VectorOfIntType:$indices);
  let results = (outs CIR_VectorType:$result);
  let assemblyFormat = [{
    $vec `:` qualified(type($vec)) `,` $indices `:` qualified(type($indices))
    attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VecTernaryOp
//===----------------------------------------------------------------------===//

def VecTernaryOp : CIR_Op<"vec.ternary",
                   [Pure, AllTypesMatch<["result", "lhs", "rhs"]>]> {
  let summary = "The `cond ? a : b` ternary operator for vector types";
  let description = [{
    The `cir.vec.ternary` operation represents the C/C++ ternary operator,
    `?:`, for vector types, which does a `select` on individual elements of the
    vectors. Unlike a regular `?:` operator, there is no short circuiting. All
    three arguments are always evaluated.  Because there is no short
    circuiting, there are no regions in this operation, unlike cir.ternary.

    The first argument is a vector of integral type. The second and third
    arguments are vectors of the same type and have the same number of elements
    as the first argument.

    The result is a vector of the same type as the second and third arguments.
    Each element of the result is `(bool)a[n] ? b[n] : c[n]`.
  }];

  let arguments = (ins
    CIR_VectorOfIntType:$cond,
    CIR_VectorType:$lhs,
    CIR_VectorType:$rhs
  );

  let results = (outs CIR_VectorType:$result);
  let assemblyFormat = [{
    `(` $cond `,` $lhs`,` $rhs `)` `:` qualified(type($cond)) `,`
    qualified(type($lhs)) attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VecSplatOp
//===----------------------------------------------------------------------===//

def VecSplatOp : CIR_Op<"vec.splat", [Pure,
  TypesMatchWith<"type of 'value' matches element type of 'result'", "result",
                 "value", "cast<VectorType>($_self).getElementType()">]> {

  let summary = "Convert a scalar into a vector";
  let description = [{
    The `cir.vec.splat` operation creates a vector value from a scalar value.
    All elements of the vector have the same value, that of the given scalar.

    It's a separate operation from `cir.vec.create` because more
    efficient LLVM IR can be generated for it, and because some optimization and
    analysis passes can benefit from knowing that all elements of the vector
    have the same value.

    ```mlir
    %value = cir.const #cir.int<3> : !s32i
    %value_vec = cir.vec.splat %value : !s32i, !cir.vector<4 x !s32i>
    ```
  }];

  let arguments = (ins CIR_VectorElementType:$value);
  let results = (outs CIR_VectorType:$result);

  let assemblyFormat = [{
    $value `:` type($value) `,` qualified(type($result)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BaseClassAddrOp
//===----------------------------------------------------------------------===//

def BaseClassAddrOp : CIR_Op<"base_class_addr"> {
  let summary = "Get the base class address for a class/struct";
  let description = [{
    The `cir.base_class_addr` operaration gets the address of a particular
    non-virtual base class given a derived class pointer. The offset in bytes
    of the base class must be passed in, since it is easier for the front end
    to calculate that than the MLIR passes. The operation contains a flag for
    whether or not the operand may be nullptr. That depends on the context and
    cannot be known by the operation, and that information affects how the
    operation is lowered.

    Example:
    ```c++
    struct Base { };
    struct Derived : Base { };
    Derived d;
    Base& b = d;
    ```
    will generate
    ```mlir
    %3 = cir.base_class_addr %1 : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
    ```
  }];

  // The validity of the relationship of derived and base cannot yet be
  // verified, currently not worth adding a verifier.
  let arguments = (ins
    Arg<CIR_PointerType, "derived class pointer", [MemRead]>:$derived_addr,
    IndexAttr:$offset, UnitAttr:$assume_not_null);

  let results = (outs Res<CIR_PointerType, "">:$base_addr);

  let assemblyFormat = [{
      $derived_addr `:` qualified(type($derived_addr))
      (`nonnull` $assume_not_null^)?
      ` ` `[` $offset `]` `->` qualified(type($base_addr)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ComplexCreateOp
//===----------------------------------------------------------------------===//

def ComplexCreateOp : CIR_Op<"complex.create", [Pure, SameTypeOperands]> {
  let summary = "Create a complex value from its real and imaginary parts";
  let description = [{
    The `cir.complex.create` operation takes two operands that represent the
    real and imaginary part of a complex number, and yields the complex number.

    ```mlir
    %0 = cir.const #cir.fp<1.000000e+00> : !cir.double
    %1 = cir.const #cir.fp<2.000000e+00> : !cir.double
    %2 = cir.complex.create %0, %1 : !cir.double -> !cir.complex<!cir.double>
    ```
  }];

  let results = (outs CIR_ComplexType:$result);
  let arguments = (ins
    CIR_AnyIntOrFloatType:$real,
    CIR_AnyIntOrFloatType:$imag
  );

  let assemblyFormat = [{
    $real `,` $imag
    `:` qualified(type($real)) `->` qualified(type($result)) attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ComplexRealOp
//===----------------------------------------------------------------------===//

def ComplexRealOp : CIR_Op<"complex.real", [Pure]> {
  let summary = "Extract the real part of a complex value";
  let description = [{
    `cir.complex.real` operation takes an operand of `!cir.complex` type and
    yields the real part of it.

    Example:

    ```mlir
    %1 = cir.complex.real %0 : !cir.complex<!cir.float> -> !cir.float
    ```
  }];

  let results = (outs CIR_AnyIntOrFloatType:$result);
  let arguments = (ins CIR_ComplexType:$operand);

  let assemblyFormat = [{
    $operand `:` qualified(type($operand)) `->` qualified(type($result))
    attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ComplexImagOp
//===----------------------------------------------------------------------===//

def ComplexImagOp : CIR_Op<"complex.imag", [Pure]> {
  let summary = "Extract the imaginary part of a complex value";
  let description = [{
    `cir.complex.imag` operation takes an operand of `!cir.complex` type and
    yields the imaginary part of it.

    Example:

    ```mlir
    %1 = cir.complex.imag %0 : !cir.complex<!cir.float> -> !cir.float
    ```
  }];

  let results = (outs CIR_AnyIntOrFloatType:$result);
  let arguments = (ins CIR_ComplexType:$operand);

  let assemblyFormat = [{
    $operand `:` qualified(type($operand)) `->` qualified(type($result))
    attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Bit Manipulation Operations
//===----------------------------------------------------------------------===//

class CIR_BitOpBase<string mnemonic, TypeConstraint operandTy>
  : CIR_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins operandTy:$input);
  let results = (outs operandTy:$result);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` `:` type($result) attr-dict
  }];
}

class CIR_BitZeroCountOpBase<string mnemonic, TypeConstraint operandTy>
  : CIR_BitOpBase<mnemonic, operandTy> {
  let arguments = (ins operandTy:$input, UnitAttr:$poison_zero);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` (`poison_zero` $poison_zero^)?
    `:` type($result) attr-dict
  }];
}

def BitClrsbOp : CIR_BitOpBase<"bit.clrsb", CIR_SIntOfWidths<[32, 64]>> {
  let summary = "Get the number of leading redundant sign bits in the input";
  let description = [{
    Compute the number of leading redundant sign bits in the input integer.

    The input integer must be a signed integer. The most significant bit of the
    input integer is the sign bit. The `cir.bit.clrsb` operation returns the
    number of consecutive bits following the sign bit that are identical to the
    sign bit.

    The bit width of the input integer must be either 32 or 64.

    Examples:

    ```mlir
    // %0 = 0b1101_1110_1010_1101_1011_1110_1110_1111
    %0 = cir.const #cir.int<3735928559> : !s32i
    // %1 will be 1 because there is 1 bit following the most significant bit
    // that is identical to it.
    %1 = cir.bit.clrsb(%0 : !s32i) : !s32i

    // %2 = 1, 0b0000_0000_0000_0000_0000_0000_0000_0001
    %2 = cir.const #cir.int<1> : !s32i
    // %3 will be 30 because there are 30 consecutive bits following the sign
    // bit that are identical to the sign bit.
    %3 = cir.bit.clrsb(%2 : !s32i) : !s32i
    ```
  }];
}

def BitClzOp : CIR_BitZeroCountOpBase<"bit.clz",
                                      CIR_UIntOfWidths<[16, 32, 64]>> {
  let summary = "Get the number of leading 0-bits in the input";
  let description = [{
    Compute the number of leading 0-bits in the input.

    The input integer must be an unsigned integer. The `cir.bit.clz` operation
    returns the number of consecutive 0-bits at the most significant bit
    position in the input.

    If the `poison_zero` attribute is present, this operation will have
    undefined behavior if the input value is 0.

    Example:

    ```mlir
    // %0 = 0b0000_0000_0000_0000_0000_0000_0000_1000
    %0 = cir.const #cir.int<8> : !u32i
    // %1 will be 28
    %1 = cir.bit.clz(%0 : !u32i) poison_zero : !u32i
    ```
  }];
}

def BitCtzOp : CIR_BitZeroCountOpBase<"bit.ctz",
                                      CIR_UIntOfWidths<[16, 32, 64]>> {
  let summary = "Get the number of trailing 0-bits in the input";
  let description = [{
    Compute the number of trailing 0-bits in the input.

    The input integer must be an unsigned integer. The `cir.bit.ctz` operation
    counts the number of consecutive 0-bits starting from the least significant
    bit.

    If the `poison_zero` attribute is present, this operation will have
    undefined behavior if the input value is 0.

    Example:

    ```mlir
    // %0 = 0b1000
    %0 = cir.const #cir.int<8> : !u32i
    // %1 will be 3
    %1 = cir.bit.ctz(%0 : !u32i) poison_zero : !u32i
    ```
  }];
}

def BitParityOp : CIR_BitOpBase<"bit.parity", CIR_UIntOfWidths<[32, 64]>> {
  let summary = "Get the parity of input";
  let description = [{
    Compute the parity of the input. The parity of an integer is the number of
    1-bits in it modulo 2.

    The input must be an unsigned integer.

    Example:

    ```mlir
    // %0 = 0x0110_1000
    %0 = cir.const #cir.int<104> : !u32i
    // %1 will be 1 since there are three 1-bits in %0
    %1 = cir.bit.parity(%0 : !u32i) : !u32i
    ```
  }];
}

def BitPopcountOp : CIR_BitOpBase<"bit.popcnt",
                                  CIR_UIntOfWidths<[16, 32, 64]>> {
  let summary = "Get the number of 1-bits in input";
  let description = [{
    Compute the number of 1-bits in the input.

    The input must be an unsigned integer.

    Example:

    ```mlir
    // %0 = 0x0110_1000
    %0 = cir.const #cir.int<104> : !u32i
    // %1 will be 3 since there are 3 1-bits in %0
    %1 = cir.bit.popcnt(%0 : !u32i) : !u32i
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Assume Operations
//===----------------------------------------------------------------------===//

def AssumeOp : CIR_Op<"assume"> {
  let summary = "Tell the optimizer that a boolean value is true";
  let description = [{
    The `cir.assume` operation takes a single boolean prediate as its only
    argument and does not have any results. The operation tells the optimizer
    that the predicate is always true.

    This operation corresponds to the `__assume` and the `__builtin_assume`
    builtin functions.
  }];

  let arguments = (ins CIR_BoolType:$predicate);

  let assemblyFormat = [{
    $predicate `:` type($predicate) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Branch Probability Operations
//===----------------------------------------------------------------------===//

def ExpectOp : CIR_Op<"expect",
  [Pure, AllTypesMatch<["result", "val", "expected"]>]> {
  let summary = "Tell the optimizer that two values are likely to be equal.";
  let description = [{
    The `cir.expect` operation may take 2 or 3 arguments.

    When the argument `prob` is missing, this operation effectively models the
    `__builtin_expect` builtin function. It tells the optimizer that `val` and
    `expected` are likely to be equal.

    When the argument `prob` is present, this operation effectively models the
    `__builtin_expect_with_probability` builtin function. It tells the
    optimizer that `val` and `expected` are equal to each other with a certain
    probability.

    `val` and `expected` must be integers and their types must match.

    The result of this operation is always equal to `val`.
  }];

  let arguments = (ins
    CIR_AnyFundamentalIntType:$val,
    CIR_AnyFundamentalIntType:$expected,
    OptionalAttr<F64Attr>:$prob
  );

  let results = (outs CIR_AnyFundamentalIntType:$result);

  let assemblyFormat = [{
    `(` $val`,` $expected (`,` $prob^)? `)` `:` type($val) attr-dict
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIROPS_TD
