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
    Op<CIR_Dialect, mnemonic, traits>, LLVMLoweringInfo {
  // Should we generate an ABI lowering pattern for this op?
  bit hasCXXABILowering = false;
  // Should we generate an LLVM lowering pattern for this op?
  bit hasLLVMLowering = true;
  // Is the LLVM lowering pattern for this operation recursive?
  bit isLLVMLoweringRecursive = false;
  // Extra class declarations to be included in the generated LLVM lowering
  // pattern.
  code extraLLVMLoweringPatternDecl = "";
}

//===----------------------------------------------------------------------===//
// CIR Operation Traits
//===----------------------------------------------------------------------===//

class HasAtMostOneOfAttrsPred<list<string> names> :
  CPred<!foldl("0", names, acc, name,  acc # " + (" # name # " ? 1 : 0)")
        # " <= 1">;

class HasAtMostOneOfAttrs<list<string> names> : PredOpTrait<
  "has only one of the optional attributes: " # !interleave(names, ", "),
  HasAtMostOneOfAttrsPred<!foreach(name, names, "$" # name)>
>;

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
  I32EnumAttrCase<"float_to_complex", 44>,
  I32EnumAttrCase<"float_complex_to_real", 45>,
  I32EnumAttrCase<"float_complex_to_bool", 46>,
  I32EnumAttrCase<"float_complex", 47>,
  I32EnumAttrCase<"float_complex_to_int_complex", 48>,
  I32EnumAttrCase<"int_to_complex", 49>,
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

def CIR_CastOp : CIR_Op<"cast", [
  Pure, DeclareOpInterfaceMethods<PromotableOpInterface>
]> {
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
    %4 = cir.cast int_to_bool %3 : i32 -> !cir.bool
    ...
    %x = cir.cast array_to_ptrdecay %0 
       : !cir.ptr<!cir.array<i32 x 10>> -> !cir.ptr<i32>
    ```
  }];

  let arguments = (ins CIR_CastKind:$kind, CIR_AnyType:$src);
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    $kind $src `:` type($src) `->` type($result) attr-dict
  }];

  // The input and output types should match the cast kind.
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCXXABILowering = 1;

  let extraLLVMLoweringPatternDecl = [{
    mlir::Type convertTy(mlir::Type ty) const;
  }];
}

//===----------------------------------------------------------------------===//
// DynamicCastOp
//===----------------------------------------------------------------------===//

def CIR_DynamicCastKind : CIR_I32EnumAttr<
  "DynamicCastKind", "dynamic cast kind", [
    I32EnumAttrCase<"Ptr", 0, "ptr">,
    I32EnumAttrCase<"Ref", 1, "ref">
]>;

def CIR_DynamicCastOp : CIR_Op<"dyn_cast"> {
  let summary = "Perform dynamic cast on record pointers";
  let description = [{
    The `cir.dyn_cast` operation models part of the semantics of the
    `dynamic_cast` operator in C++. It can be used to perform 3 kinds of casts
    on record pointers:

    - Down-cast, which casts a base class pointer to a derived class pointer;
    - Side-cast, which casts a class pointer to a sibling class pointer;
    - Cast-to-complete, which casts a class pointer to a void pointer.

    The input of the operation must be a record pointer. The result of the
    operation is either a record pointer or a void pointer.

    The parameter `kind` specifies the semantics of this operation. If its value
    is `ptr`, then the operation models dynamic casts on pointers. Otherwise, if
    its value is `ref`, the operation models dynamic casts on references.
    Specifically:

    - When the input pointer is a null pointer value:
      - If `kind` is `ref`, the operation will invoke undefined behavior. A
        sanitizer check will be emitted if sanitizer is on.
      - Otherwise, the operation will return a null pointer value as its result.
    - When the runtime type check fails:
      - If `kind` is `ref`, the operation will throw a `bad_cast` exception.
      - Otherwise, the operation will return a null pointer value as its result.

    The `info` argument gives detailed information about the requested dynamic
    cast operation. It is an optional `#cir.dyn_cast_info` attribute that is
    only present when the operation models a down-cast or a side-cast.

    The `relative_layout` argument specifies whether the Itanium C++ ABI vtable
    uses relative layout. It is only meaningful when the operation models a
    cast-to-complete operation.

    Examples:

    ```mlir
    %0 = cir.dyn_cast ptr %p : !cir.ptr<!rec_Base> -> !cir.ptr<!rec_Derived>
    %1 = cir.dyn_cast ptr relative_layout %p : !cir.ptr<!rec_Base>
              -> !cir.ptr<!rec_Derived>
    %2 = cir.dyn_cast ref %r : !cir.ptr<!rec_Base> -> !cir.ptr<!rec_Derived>
              #cir.dyn_cast_info<
                srcRtti = #cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>,
                destRtti = #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>,
                runtimeFunc = @__dynamic_cast,
                badCastFunc = @__cxa_bad_cast,
                offsetHint = #cir.int<0> : !s64i
              >
    ```
  }];

  let arguments = (ins
    CIR_DynamicCastKind:$kind,
    CIR_PtrToRecordType:$src,
    OptionalAttr<CIR_DynamicCastInfoAttr>:$info,
    UnitAttr:$relative_layout
  );

  let results = (outs
    CIR_PtrToAnyOf<[CIR_VoidType, CIR_RecordType]>:$result
  );

  let assemblyFormat = [{
    $kind (`relative_layout` $relative_layout^)? $src
    `:` qualified(type($src)) `->` qualified(type($result))
    (qualified($info)^)? attr-dict
  }];

  let extraClassDeclaration = [{
    /// Determine whether this operation models reference casting in C++.
    bool isRefCast() {
      return getKind() == ::cir::DynamicCastKind::Ref;
    }

    /// Determine whether this operation represents a dynamic cast to a void
    /// pointer.
    bool isCastToVoid() {
      return getType().isVoidPtr();
    }
  }];

  let hasLLVMLowering = false;
  let hasCXXABILowering = true;
}

//===----------------------------------------------------------------------===//
// PtrStrideOp
//===----------------------------------------------------------------------===//

def CIR_PtrStrideOp : CIR_Op<"ptr_stride", [
  Pure, AllTypesMatch<["base", "result"]>
]> {
  let summary = "Pointer access with stride";
  let description = [{
    The `cir.ptr_stride` operation computes a new pointer from a base pointer
    and an integer stride, similar to a single-index `getelementptr` in LLVM IR.
    It moves the pointer by `stride * sizeof(element_type)` bytes.

    ```mlir
    %3 = cir.const 0 : i32
    %3 = cir.ptr_stride %1, %2 : (!cir.ptr<i32>, i32) -> !cir.ptr<i32>
    ```
  }];

  let arguments = (ins
    CIR_PointerType:$base,
    CIR_AnyFundamentalIntType:$stride
  );

  let results = (outs CIR_PointerType:$result);

  let assemblyFormat = [{
    $base`,` $stride `:` functional-type(operands, results) attr-dict
  }];

  let extraClassDeclaration = [{
    // Get type pointed by the base pointer.
    mlir::Type getElementType() {
      return getBase().getType().getPointee();
    }
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def CIR_ConstantOp : CIR_Op<"const", [
  ConstantLike, Pure, AllTypesMatch<["value", "res"]>
]> {
  let summary = "Create a CIR constant from a literal attribute";
  let description = [{
    The `cir.const` operation turns a literal into an SSA value. The data is
    attached to the operation as an attribute.

    ```mlir
      %0 = cir.const #cir.int<4> : !u32i
      %1 = cir.const #cir.fp<1.500000e+00> : !cir.float
      %2 = cir.const #cir.ptr<null> : !cir.ptr<!void>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs CIR_AnyType:$res);

  let assemblyFormat = "$value attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isNullPtr() {
      if (const auto ptrAttr = mlir::dyn_cast<cir::ConstPtrAttr>(getValue()))
        return ptrAttr.isNullValue();
      return false;
    }

    template <typename T>
    T getValueAttr() { return mlir::dyn_cast<T>(getValue()); }

    llvm::APInt getIntValue() {
      if (const auto intAttr = getValueAttr<cir::IntAttr>())
        return intAttr.getValue();
      llvm_unreachable("Expected an IntAttr in ConstantOp");
    }

    llvm::APFloat getFloatValue() {
      if (const auto fpAttr = getValueAttr<cir::FPAttr>())
        return fpAttr.getValue();
      llvm_unreachable("Expected an FPAttr in ConstantOp");
    }

    bool getBoolValue() {
      if (const auto boolAttr = getValueAttr<cir::BoolAttr>())
        return boolAttr.getValue();
      llvm_unreachable("Expected a BoolAttr in ConstantOp");
    }

    bool isAllOnesValue() {
      // Check for -1 integers
      if (auto intAttr = getValueAttr<cir::IntAttr>())
        return intAttr.getValue().isAllOnes();

      // Check for FP which are bitcasted from -1 integers
      if (auto fpAttr = getValueAttr<cir::FPAttr>())
        return fpAttr.getValue().bitcastToAPInt().isAllOnes();

      // Check for constant vectors with splat values
      if (cir::VectorType v = mlir::dyn_cast<cir::VectorType>(getType()))
        if (auto vecAttr = getValueAttr<mlir::DenseElementsAttr>())
          if (vecAttr.isSplat()) {
            auto splatAttr = vecAttr.getSplatValue<mlir::Attribute>();
            if (auto splatInt = mlir::dyn_cast<cir::IntAttr>(splatAttr)) {
              return splatInt.getValue().isAllOnes();
          }
      }

      return false;
    }
  }];

  let hasFolder = 1;

  let hasCXXABILowering = true;
  let isLLVMLoweringRecursive = true;
}

//===----------------------------------------------------------------------===//
// C/C++ memory order definitions
//===----------------------------------------------------------------------===//

def CIR_MemOrder : CIR_I32EnumAttr<
  "MemOrder", "Memory order according to C++11 memory model", [
    I32EnumAttrCase<"Relaxed", 0, "relaxed">,
    I32EnumAttrCase<"Consume", 1, "consume">,
    I32EnumAttrCase<"Acquire", 2, "acquire">,
    I32EnumAttrCase<"Release", 3, "release">,
    I32EnumAttrCase<"AcquireRelease", 4, "acq_rel">,
    I32EnumAttrCase<"SequentiallyConsistent", 5, "seq_cst">
]>;

//===----------------------------------------------------------------------===//
// C/C++ sync scope definitions
//===----------------------------------------------------------------------===//

def CIR_SyncScopeKind : CIR_I32EnumAttr<"SyncScopeKind", "sync scope kind", [
  I32EnumAttrCase<"SingleThread", 0, "single_thread">,
  I32EnumAttrCase<"System", 1, "system">
]>;

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

class CIR_AllocaTypesMatchWith<
  string summary, string lhsArg, string rhsArg, string transform,
  string comparator = "std::equal_to<>()"
> : PredOpTrait<summary, CPred<comparator # "(" #
      !subst("$_self", "$" # lhsArg # ".getType()", transform) #
             ", $" # rhsArg # ")">
> {
  string lhs = lhsArg;
  string rhs = rhsArg;
  string transformer = transform;
}

def CIR_AllocaOp : CIR_Op<"alloca", [
  CIR_AllocaTypesMatchWith<"'allocaType' matches pointee type of 'addr'",
    "addr", "allocaType", "mlir::cast<cir::PointerType>($_self).getPointee()">,
  DeclareOpInterfaceMethods<PromotableAllocationOpInterface>
]> {
  let summary = "Defines a scope-local variable";
  let description = [{
    The `cir.alloca` operation defines a scope-local variable.

    The presence of the `const` attribute indicates that the local variable is
    declared with C/C++ `const` keyword.

    The `dynAllocSize` specifies the size to dynamically allocate on the stack
    and ignores the allocation size based on the original type. This is useful
    when handling VLAs or the `alloca` builtin and is omitted when declaring
    regular local variables.

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
    Optional<CIR_AnyFundamentalIntType>:$dynAllocSize,
    TypeAttr:$allocaType,
    StrAttr:$name,
    UnitAttr:$init,
    UnitAttr:$constant,
    ConfinedAttr<I64Attr, [IntMinValue<1>]>:$alignment,
    OptionalAttr<ArrayAttr>:$annotations
  );

  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$addr,
                   "mlir::Type":$allocaType,
                   "llvm::StringRef":$name,
                   "mlir::IntegerAttr":$alignment)>,

    OpBuilder<(ins "mlir::Type":$addr,
                   "mlir::Type":$allocaType,
                   "llvm::StringRef":$name,
                   "mlir::IntegerAttr":$alignment,
                   "mlir::Value":$dynAllocSize),
    [{
      if (dynAllocSize)
        $_state.addOperands(dynAllocSize);
      build($_builder, $_state, addr, allocaType, name, alignment);
    }]>
  ];

  let extraClassDeclaration = [{
    // Whether the alloca input type is a pointer.
    bool isPointerType() { return ::mlir::isa<::cir::PointerType>(getAllocaType()); }
    bool isDynamic() { return (bool)getDynAllocSize(); }
  }];

  let assemblyFormat = [{
    $allocaType `,` qualified(type($addr)) `,`
    ($dynAllocSize^ `:` type($dynAllocSize) `,`)?
    `[` $name
       (`,` `init` $init^)?
       (`,` `const` $constant^)?
    `]`
    ($annotations^)? attr-dict
  }];

  let hasCXXABILowering = true;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def CIR_LoadOp : CIR_Op<"load", [
  TypesMatchWith<"type of 'result' matches pointee type of 'addr'",
    "addr", "result", "mlir::cast<cir::PointerType>($_self).getPointee()">,
  DeclareOpInterfaceMethods<PromotableMemOpInterface>
]> {
  let summary = "Load value from memory adddress";
  let description = [{
    `cir.load` reads a value (lvalue to rvalue conversion) given an address
    backed up by a `cir.ptr` type. A unit attribute `deref` can be used to
    mark the resulting value as used by another operation to dereference
    a pointer. A unit attribute `volatile` can be used to indicate a volatile
    loading. Load can be marked atomic by using `atomic(<mem_order>)`.

    `alignment` can be used to specify an alignment that's different from the
    default, which is computed from `result`'s type ABI data layout.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = cir.load %0 : !cir.ptr<i32>, i32

    // Load address from memory at address %0. %3 is used by at least one
    // operation that dereferences a pointer.
    %3 = cir.load deref %0 : !cir.ptr<!cir.ptr<i32>>

    // Perform a volatile load from address in %0.
    %4 = cir.load volatile %0 : !cir.ptr<i32>, i32

    // Others
    %x = cir.load align(16) atomic(seq_cst) %0 : !cir.ptr<i32>, i32
    ```
  }];

  let arguments = (ins Arg<CIR_PointerType, "the address to load from",
                           [MemRead]>:$addr,
                       UnitAttr:$isDeref,
                       UnitAttr:$is_volatile,
                       OptionalAttr<I64Attr>:$alignment,
                       OptionalAttr<CIR_SyncScopeKind>:$sync_scope,
                       OptionalAttr<CIR_MemOrder>:$mem_order);
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    (`deref` $isDeref^)?
    (`volatile` $is_volatile^)?
    (`align` `(` $alignment^ `)`)?
    (`syncscope` `(` $sync_scope^ `)`)?
    (`atomic` `(` $mem_order^ `)`)?
    $addr `:` qualified(type($addr)) `,` type($result) attr-dict
  }];

  // FIXME: add verifier.
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def CIR_StoreOp : CIR_Op<"store", [
  TypesMatchWith<"type of 'value' matches pointee type of 'addr'",
    "addr", "value", "mlir::cast<cir::PointerType>($_self).getPointee()">,
  DeclareOpInterfaceMethods<PromotableMemOpInterface>
]> {
  let summary = "Store value to memory address";
  let description = [{
    `cir.store` stores a value (first operand) to the memory address specified
    in the second operand. A unit attribute `volatile` can be used to indicate
    a volatile store. Store's can be marked atomic by using
    `atomic(<mem_order>)`.

    `alignment` can be used to specify an alignment that's different from the
    default, which is computed from `result`'s type ABI data layout.

    Example:

    ```mlir
    // Store a function argument to local storage, address in %0.
    cir.store %arg0, %0 : i32, !cir.ptr<i32>

    // Perform a volatile store into memory location at the address in %0.
    cir.store volatile %arg0, %0 : i32, !cir.ptr<i32>

    // Others
    cir.store align(16) atomic(seq_cst) %x, %addr : i32, !cir.ptr<i32>
    ```
  }];

  let arguments = (ins CIR_AnyType:$value,
                       Arg<CIR_PointerType, "the address to store the value",
                           [MemWrite]>:$addr,
                       UnitAttr:$is_volatile,
                       OptionalAttr<I64Attr>:$alignment,
                       OptionalAttr<CIR_SyncScopeKind>:$sync_scope,
                       OptionalAttr<CIR_MemOrder>:$mem_order);

  let assemblyFormat = [{
    (`volatile` $is_volatile^)?
    (`align` `(` $alignment^ `)`)?
    (`syncscope` `(` $sync_scope^ `)`)?
    (`atomic` `(` $mem_order^ `)`)?
    $value `,` $addr attr-dict `:` type($value) `,` qualified(type($addr))
  }];

  // FIXME: add verifier.
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

defvar CIR_ReturnableScopes = [
  "FuncOp", "ScopeOp", "IfOp", "SwitchOp", "CaseOp",
  "DoWhileOp", "WhileOp", "ForOp", "TryOp"
];

def CIR_ReturnOp : CIR_Op<"return", [
  ParentOneOf<CIR_ReturnableScopes>, Terminator
]> {
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

def CIR_IfOp : CIR_Op<"if", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments
]> {
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

  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

def CIR_ConditionOp : CIR_Op<"condition", [
  Terminator,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, [
    "getSuccessorRegions"
  ]>
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
  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

defvar CIR_YieldableScopes = [
  "ArrayCtor", "ArrayDtor", "AwaitOp", "CaseOp", "DoWhileOp", "ForOp",
  "GlobalOp", "IfOp", "ScopeOp", "SwitchOp", "TernaryOp", "WhileOp", "TryOp"
];

def CIR_YieldOp : CIR_Op<"yield", [
  ReturnLike, Terminator, ParentOneOf<CIR_YieldableScopes>, NoMemoryEffect
]> {
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

  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

def CIR_BreakOp : CIR_Op<"break", [Terminator]> {
  let summary = "C/C++ `break` statement equivalent";
  let description = [{
    The `cir.break` operation is used to cease the execution of the current loop
    or switch operation and transfer control to the parent operation. It is only
    allowed within a breakable operations (loops and switches).
  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

def CIR_ContinueOp : CIR_Op<"continue", [Terminator]> {
  let summary = "C/C++ `continue` statement equivalent";
  let description = [{
    The `cir.continue` operation is used to end execution of the current
    iteration of a loop and resume execution beginning at the next iteration.
    It is only allowed within loop regions.
  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// Resume
//===----------------------------------------------------------------------===//

def CIR_ResumeOp : CIR_Op<"resume", [
  ReturnLike, Terminator, HasParent<"cir::TryOp">
]> {
  let summary = "Resumes execution after not catching exceptions";
  let description = [{
    The `cir.resume` operation handles an uncaught exception scenario.

    Used as the terminator of a `CatchUnwind` region of `cir.try`, where it
    does not receive any arguments (implied from the `cir.try` scope).

    This operation is used only before the CFG flatterning pass.

    Examples:
    ```mlir
    cir.try {
      cir.yield
    } unwind {
      cir.resume
    }
    ```
  }];

  let assemblyFormat = "attr-dict";
  let hasLLVMLowering = false;
}

def CIR_ResumeFlatOp : CIR_Op<"resume.flat", [
  ReturnLike, Terminator
]> {
  let summary = "A flattened version of `cir.resume`";
  let description = [{
    The `cir.resume.flat` operation is a region-less and simplified
    version of the `cir.resume`.

    Its representation is closer to LLVM IR dialect
    than the C/C++ language feature.

    This operation is used only after the CFG flatterning pass.

    Examples:
    ```mlir
    cir.resume.flat %exception_ptr, %type_id
    ```
  }];

  let arguments = (ins
    CIR_VoidPtrType:$exception_ptr,
    CIR_UInt32:$type_id
  );

  let assemblyFormat = [{
    $exception_ptr `,` $type_id
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

def CIR_ScopeOp : CIR_Op<"scope", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments,
  RecursiveMemoryEffects
]> {
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

  let hasFolder = 1;
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

  let hasLLVMLowering = false;
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

def CIR_CaseOp : CIR_Op<"case", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  RecursivelySpeculatable, AutomaticAllocationScope
]> {
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

  let hasLLVMLowering = false;
}

def CIR_SwitchOp : CIR_Op<"switch", [
  SameVariadicOperandSize,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments,
  RecursiveMemoryEffects
]> {
  let summary = "Switch operation";
  let description = [{
    The `cir.switch` operation represents C/C++ switch functionality for
    conditionally executing multiple regions of code. The operand to an switch
    is an integral condition value.

    Besides taking an integer condition and CIR regions, it also accepts an
    `all_enum_cases_covered` attribute indicating whether all enum cases are 
    handled by the operation. Note that the presence of a default CaseOp does 
    not imply `all_enum_cases_covered`. The original AST switch must explicitly list 
    every enum case.

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

  let arguments = (ins 
    CIR_IntType:$condition,
    UnitAttr:$all_enum_cases_covered
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$condition,
               "BuilderOpStateCallbackRef":$switchBuilder)>
  ];

  let assemblyFormat = [{
    `(` $condition `:` qualified(type($condition)) `)` 
     (`all_enum_cases_covered` $all_enum_cases_covered^)?
    $body
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

  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// IsConstantOp
//===----------------------------------------------------------------------===//

def CIR_IsConstantOp : CIR_Op<"is_constant", [Pure]> {
  let summary = "Test for manifest compile-time constant";
  let description = [{
    Returns `true` if the argument is known to be a manifest compile-time
    constant otherwise returns `false`. If the argument is a constant expression
    which refers to a global (the address of which _is_ a constant, but not
    manifest during the compile), then the intrinsic evaluates to `false`.

    This is used to represent `__builtin_constant_p` in cases where the argument
    isn't known to be constant during initial translation of the source code but
    might be proven to be constant after later optimizations.

    Example:
    ```
    %1 = cir.is_constant %2 : !s32i -> !cir.bool
    ```
  }];
  let arguments = (ins CIR_AnyType:$val);
  let results = (outs CIR_BoolType:$result);

  let assemblyFormat = [{
    $val `:` qualified(type($val)) `->` qualified(type($result)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// SwitchFlatOp
//===----------------------------------------------------------------------===//

def CIR_SwitchFlatOp : CIR_Op<"switch.flat", [
  AttrSizedOperandSegments, Terminator
]> {
  let summary = "A flattened version of cir.switch";

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

def CIR_BrOp : CIR_Op<"br",[
  DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
  Pure, Terminator
]> {
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
// GotoOp
//===----------------------------------------------------------------------===//

def CIR_GotoOp : CIR_Op<"goto", [Terminator]> {
  let description = [{

  Transfers control to the specified `label`. This requires a corresponding
  `cir.label` to exist and is used by to represent source level `goto`s
  that jump across region boundaries. Alternatively, `cir.br` is used to
  construct goto's that don't violate such boundaries.

  `cir.goto` is completely symbolic (i.e. it "jumps" on a label that isn't
  yet materialized) and should be taken into account by passes and analysis
  when deciding if it's safe to make some assumptions about a given region
  or basic block.

  Example:
  ```C++
    int test(int x) {
      if (x)
        goto label;
      {
        x = 10;
    label:
        return x;
      }
    }
  ```

  ```mlir
    cir.scope {  // REGION #1
      %2 = cir.load %0 : !cir.ptr<!s32i>, !s32i
      %3 = cir.cast int_to_bool %2 : !s32i -> !cir.bool
      cir.if %3 {
        cir.goto "label"
      }
      }
      cir.scope {  // REGION #2
        %2 = cir.const #cir.int<10> : !s32i
        cir.store %2, %0 : !s32i, !cir.ptr<!s32i>
        cir.br ^bb1
      ^bb1:  // pred: ^bb0
        cir.label "label"
        %3 = cir.load %0 : !cir.ptr<!s32i>, !s32i
        cir.store %3, %1 : !s32i, !cir.ptr<!s32i>
        %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
        cir.return %4 : !s32i
      }
      cir.unreachable
  ```
  }];
  let arguments = (ins StrAttr:$label);
  let assemblyFormat = [{ $label attr-dict }];

  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// LabelOp
//===----------------------------------------------------------------------===//

// The LabelOp has AlwaysSpeculatable trait in order to not to be swept
// by canonicalizer
def CIR_LabelOp : CIR_Op<"label", [AlwaysSpeculatable]> {
  let description = [{
    An identifier which may be referred by cir.goto operation
  }];
  let arguments = (ins StrAttr:$label);
  let assemblyFormat = [{ $label attr-dict }];
  let hasVerifier = 1;

  let hasLLVMLowering = false;
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

def CIR_UnaryOp : CIR_Op<"unary", [Pure, SameOperandsAndResultType]> {
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

def CIR_BrCondOp : CIR_Op<"brcond", [
  DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
  Pure, Terminator, AttrSizedOperandSegments
]> {
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
// IndirectBrOp
//===----------------------------------------------------------------------===//

def CIR_IndirectBrOp : CIR_Op<"indirect_br", [
  DeclareOpInterfaceMethods<BranchOpInterface>,
  SameVariadicOperandSize, Terminator, Pure
]> {
  let summary = "Indirect branch";
  let description = [{
    The `cir.indirectbr` operation represents an indirect branch to one of
    several possible successor blocks. The target block is computed from
    the value of the given address operand.

    This operation is typically generated when handling constructs like
    the GCC extension `&&label` combined with an indirect `goto *ptr;`.

    The `poison` attribute is used to mark an `indirectbr` that was created
    but is known to be invalid, for instance when a label address was
    taken but no indirect branch was ever emitted.

    Example:

    ```mlir
      %0 = cir.block_address <@A, "A"> : !cir.ptr<!void>
      cir.indirectbr %0 poison : <!void>, [
      ^bb1
      ]
    ```
  }];

  let arguments = (ins
    CIR_VoidPtrType:$addr,
    UnitAttr:$poison,
    VariadicOfVariadic<AnyType, "operand_segments">:$succ_operands,
    DenseI32ArrayAttr:$operand_segments
    );

  let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);
  let assemblyFormat = [{
    $addr ( `poison` $poison^ )? `:` qualified(type($addr)) `,`
    custom<IndirectBrOpSucessors>(ref(type($addr)),
                                  $successors,
                                  $succ_operands,
                                  type($succ_operands))
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Common loop op definitions
//===----------------------------------------------------------------------===//

class CIR_LoopOpBase<string mnemonic> : CIR_Op<mnemonic, [
  LoopOpInterface, NoRegionArguments
]> {
  let extraClassDefinition = [{
    void $cppClass::getSuccessorRegions(
        mlir::RegionBranchPoint point,
        llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
      LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
    }
    ValueRange $cppClass::getSuccessorInputs(
        mlir::RegionSuccessor successor) {
      return LoopOpInterface::getLoopOpSuccessorInputs(*this, successor);
    }
    llvm::SmallVector<Region *> $cppClass::getLoopRegions() {
      return {&getBody()};
    }
  }];
}

//===----------------------------------------------------------------------===//
// While & DoWhileOp
//===----------------------------------------------------------------------===//

class CIR_WhileOpBase<string mnemonic> : CIR_LoopOpBase<mnemonic> {
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

def CIR_WhileOp : CIR_WhileOpBase<"while"> {
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

  let hasLLVMLowering = false;
}

def CIR_DoWhileOp : CIR_WhileOpBase<"do"> {
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

  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

def CIR_ForOp : CIR_LoopOpBase<"for"> {
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

  let hasLLVMLowering = false;
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

def CIR_CmpOp : CIR_Op<"cmp", [Pure, SameTypeOperands]> {
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

  let isLLVMLoweringRecursive = true;
  let hasCXXABILowering = true;
}

//===----------------------------------------------------------------------===//
// BinOpOverflowOp
//===----------------------------------------------------------------------===//

def CIR_BinOpOverflowKind : CIR_I32EnumAttr<
  "BinOpOverflowKind", "checked binary arithmetic operation kind", [
    I32EnumAttrCase<"Add", 0, "add">,
    I32EnumAttrCase<"Sub", 1, "sub">,
    I32EnumAttrCase<"Mul", 2, "mul">
]>;

def CIR_BinOpOverflowOp : CIR_Op<"binop.overflow", [Pure, SameTypeOperands]> {
  let summary = "Perform binary integral arithmetic with overflow checking";
  let description = [{
    `cir.binop.overflow` performs binary arithmetic operations with overflow
    checking on integral operands.

    The `kind` argument specifies the kind of arithmetic operation to perform.
    It can be either `add`, `sub`, or `mul`. The `lhs` and `rhs` arguments
    specify the input operands of the arithmetic operation. The types of `lhs`
    and `rhs` must be the same.

    `cir.binop.overflow` produces two SSA values. `result` is the result of the
    arithmetic operation truncated to its specified type. `overflow` is a
    boolean value indicating whether overflow happens during the operation.

    The exact semantic of this operation is as follows:

      - `lhs` and `rhs` are promoted to an imaginary integral type that has
        infinite precision.
      - The arithmetic operation is performed on the promoted operands.
      - The infinite-precision result is truncated to the type of `result`. The
        truncated result is assigned to `result`.
      - If the truncated result is equal to the un-truncated result, `overflow`
        is assigned to false. Otherwise, `overflow` is assigned to true.
  }];

  let arguments = (ins
    CIR_BinOpOverflowKind:$kind,
    CIR_IntType:$lhs,
    CIR_IntType:$rhs
  );

  let results = (outs CIR_IntType:$result, CIR_BoolType:$overflow);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs `)` `:` qualified(type($lhs)) `,`
    `(` qualified(type($result)) `,` qualified(type($overflow)) `)`
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "cir::IntType":$resultTy,
                   "cir::BinOpOverflowKind":$kind,
                   "mlir::Value":$lhs,
                   "mlir::Value":$rhs), [{
      auto overflowTy = cir::BoolType::get($_builder.getContext());
      build($_builder, $_state, resultTy, overflowTy, kind, lhs, rhs);
    }]>
  ];

  let extraLLVMLoweringPatternDecl = [{
    static std::string getLLVMIntrinName(cir::BinOpOverflowKind opKind,
                                         bool isSigned, unsigned width);

    struct EncompassedTypeInfo {
      bool sign;
      unsigned width;
    };

    static EncompassedTypeInfo computeEncompassedTypeWidth(cir::IntType operandTy,
                                                           cir::IntType resultTy);
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

def CIR_BinOp : CIR_Op<"binop", [
  Pure, SameTypeOperands, SameOperandsAndResultType
]> {
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

  let extraLLVMLoweringPatternDecl = [{
    mlir::LLVM::IntegerOverflowFlags getIntOverflowFlag(cir::BinOp op) const;
  }];
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

def CIR_ShiftOp : CIR_Op<"shift", [Pure]> {
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

def CIR_SelectOp : CIR_Op<"select", [
  Pure, AllTypesMatch<["true_value", "false_value", "result"]>
]> {
  let summary = "Yield one of two values based on a boolean value";
  let description = [{
    The `cir.select` operation takes three operands. The first operand
    `condition` is either a boolean value of type `!cir.bool` or a boolean
    vector of type `!cir.bool`.  The second and the third operand can be of
    any CIR types, but their types must be the same. If the first operand 
    is `true`, the operation yields its second operand. Otherwise, the
    operation yields its third operand.

    In the case where the first operand is a boolean vector, then the second 
    and third operand needs to also be of some vectors of the same type to
    each other and that the number of elements of all three operands needs to 
    be the same as well.

    Example:

    ```mlir
    %0 = cir.const #cir.bool<true> : !cir.bool
    %1 = cir.const #cir.int<42> : !s32i
    %2 = cir.const #cir.int<72> : !s32i
    %3 = cir.select if %0 then %1 else %2 : (!cir.bool, !s32i, !s32i) -> !s32i
    ```
  }];

  let arguments = (ins 
    CIR_ScalarOrVectorOf<CIR_BoolType>:$condition,
    CIR_AnyType:$true_value,
    CIR_AnyType:$false_value
  );

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
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

def CIR_TernaryOp : CIR_Op<"ternary", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments
]> {
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

  let hasLLVMLowering = false;
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

def CIR_TLSModel : CIR_I32EnumAttr<"TLS_Model", "TLS model", [
  I32EnumAttrCase<"GeneralDynamic", 0, "tls_dyn">,
  I32EnumAttrCase<"LocalDynamic", 1, "tls_local_dyn">,
  I32EnumAttrCase<"InitialExec", 2, "tls_init_exec">,
  I32EnumAttrCase<"LocalExec", 3, "tls_local_exec">
]>;

def CIR_GlobalOp : CIR_Op<"global", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  DeclareOpInterfaceMethods<CIRGlobalValueInterface>,
  NoRegionArguments
]> {
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
                       OptionalAttr<CIR_TLSModel>:$tls_model,
                       OptionalAttr<AnyAttr>:$initial_value,
                       UnitAttr:$comdat,
                       UnitAttr:$constant,
                       UnitAttr:$dso_local,
                       OptionalAttr<I64Attr>:$alignment);

  let regions = (region MaxSizedRegion<1>:$ctorRegion,
                        MaxSizedRegion<1>:$dtorRegion);

  let assemblyFormat = [{
    ($sym_visibility^)?
    (`` $global_visibility^)?
    (`constant` $constant^)?
    $linkage
    (`comdat` $comdat^)?
    ($tls_model^)?
    (`dso_local` $dso_local^)?
    $sym_name
    custom<GlobalOpTypeAndInitialValue>($sym_type, $initial_value,
                                        $ctorRegion, $dtorRegion)
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool isDeclaration() {
      return !getInitialValue() && getCtorRegion().empty() && getDtorRegion().empty();
    }
    bool hasInitializer() { return !isDeclaration(); }
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$sym_name,
      "mlir::Type":$sym_type,
      CArg<"bool", "false">:$isConstant,
      // CIR defaults to external linkage.
      CArg<"cir::GlobalLinkageKind",
           "cir::GlobalLinkageKind::ExternalLinkage">:$linkage,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>",
           "nullptr">:$ctorBuilder,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>",
           "nullptr">:$dtorBuilder)
    >
  ];

  let hasVerifier = 1;

  let hasCXXABILowering = true;

  let isLLVMLoweringRecursive = true;
  let extraLLVMLoweringPatternDecl = [{
    mlir::LogicalResult matchAndRewriteRegionInitializedGlobal(
      cir::GlobalOp op, mlir::Attribute init,
      mlir::ConversionPatternRewriter &rewriter) const;

    void setupRegionInitializedLLVMGlobalOp(
        cir::GlobalOp op, mlir::ConversionPatternRewriter &rewriter) const;

    mutable mlir::LLVM::ComdatOp comdatOp = nullptr;
    mlir::SymbolRefAttr getComdatAttr(cir::GlobalOp &op,
                                      mlir::OpBuilder &builder) const;
  }];
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

def CIR_GetGlobalOp : CIR_Op<"get_global", [
  Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Get the address of a global variable";
  let description = [{
    The `cir.get_global` operation retrieves the address pointing to a
    named global variable. If the global variable is marked constant, writing
    to the resulting address (such as through a `cir.store` operation) is
    undefined. The resulting type must always be a `!cir.ptr<...>` type with the
    same address space as the global variable.

    Addresses of thread local globals can only be retrieved if this operation
    is marked `thread_local`, which indicates the address isn't constant.

    Example:
    ```mlir
    %x = cir.get_global @gv : !cir.ptr<i32>
    ...
    %y = cir.get_global thread_local @tls_gv : !cir.ptr<i32>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$name, UnitAttr:$tls);
  let results = (outs Res<CIR_PointerType, "", []>:$addr);

  let assemblyFormat = [{
    (`thread_local` $tls^)?
    $name `:` qualified(type($addr)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// VTableAddrPointOp
//===----------------------------------------------------------------------===//

def CIR_VTableAddrPointOp : CIR_Op<"vtable.address_point", [
  Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Get the vtable (global variable) address point";
  let description = [{
    The `vtable.address_point` operation retrieves the "effective" address
    (address point) of a C++ virtual table. An object internal `__vptr`
    gets initializated on top of the value returned by this operation.

    `address_point.index` (vtable index) provides the appropriate vtable within
    the vtable group (as specified by Itanium ABI), and `address_point.offset`
    (address point index) the actual address point within that vtable.

    The return type is always `!cir.vptr`.

    Example:
    ```mlir
    cir.global linkonce_odr @_ZTV1B = ...
    ...
    %3 = cir.vtable.address_point(@_ZTV1B,
              address_point = <index = 0, offset = 2>) : !cir.vptr
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$name,
    CIR_AddressPointAttr:$address_point
  );

  let results = (outs Res<CIR_VPtrType, "", []>:$addr);

  let assemblyFormat = [{
    `(`
      $name `,` `address_point` `=` $address_point
    `)`
    `:` qualified(type($addr)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// VTableGetVPtr
//===----------------------------------------------------------------------===//

def CIR_VTableGetVPtrOp : CIR_Op<"vtable.get_vptr", [Pure]> {
  let summary = "Get a the address of the vtable pointer for an object";
  let description = [{
    The `vtable.get_vptr` operation retrieves the address of the vptr for a
    C++ object. This operation requires that the object pointer points to
    the start of a complete object. (TODO: Describe how we get that).
    The vptr will always be at offset zero in the object, but this operation
    is more explicit about what is being retrieved than a direct bitcast.

    The return type is always `!cir.ptr<!cir.vptr>`.

    Example:
    ```mlir
    %2 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
    %3 = cir.vtable.get_vptr %2 : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
    ```
  }];

  let arguments = (ins
    Arg<CIR_PointerType, "the vptr address", [MemRead]>:$src
  );

  let results = (outs CIR_PtrToVPtr:$result);

  let assemblyFormat = [{
      $src `:` qualified(type($src)) `->` qualified(type($result)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// VTableGetVirtualFnAddrOp
//===----------------------------------------------------------------------===//

def CIR_VTableGetVirtualFnAddrOp : CIR_Op<"vtable.get_virtual_fn_addr", [
  Pure
]> {
  let summary = "Get a the address of a virtual function pointer";
  let description = [{
    The `vtable.get_virtual_fn_addr` operation retrieves the address of a
    virtual function pointer from an object's vtable (__vptr).
    This is an abstraction to perform the basic pointer arithmetic to get
    the address of the virtual function pointer, which can then be loaded and
    called.

    The `vptr` operand must be a `!cir.ptr<!cir.vptr>` value, which would
    have been returned by a previous call to `cir.vatble.get_vptr`. The
    `index` operand is an index of the virtual function in the vtable.

    The return type is a pointer-to-pointer to the function type.

    Example:
    ```mlir
    %2 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
    %3 = cir.vtable.get_vptr %2 : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
    %4 = cir.load %3 : !cir.ptr<!cir.vptr>, !cir.vptr
    %5 = cir.vtable.get_virtual_fn_addr %4[2] : !cir.vptr
                  -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_C>) -> !s32i>>>
    %6 = cir.load align(8) %5 : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_C>)
                                                                 -> !s32i>>>,
                                !cir.ptr<!cir.func<(!cir.ptr<!rec_C>) -> !s32i>>
    %7 = cir.call %6(%2) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_C>) -> !s32i>>,
                            !cir.ptr<!rec_C>) -> !s32i
    ```
  }];

  let arguments = (ins
    Arg<CIR_VPtrType, "vptr", [MemRead]>:$vptr,
    I64Attr:$index);

  let results = (outs CIR_PointerType:$result);

  let assemblyFormat = [{
    $vptr `[` $index `]` attr-dict
    `:` qualified(type($vptr)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// VTTAddrPointOp
//===----------------------------------------------------------------------===//

def CIR_VTTAddrPointOp : CIR_Op<"vtt.address_point", [
  Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Get the VTT address point";
  let description = [{
    The `vtt.address_point` operation retrieves an element from the virtual
    table table (VTT), which is the address point of a C++ vtable. In virtual
    inheritance, a set of internal `__vptr` members for an object are
    initialized by this operation, which assigns an element from the VTT. The
    initialization order is as follows:

    The complete object constructors and destructors find the VTT,
    via the mangled name of the VTT global variable. They pass the address of
    the subobject's sub-VTT entry in the VTT as a second parameter
    when calling the base object constructors and destructors.
    The base object constructors and destructors use the address passed to
    initialize the primary virtual pointer and virtual pointers that point to
    the classes which either have virtual bases or override virtual functions
    with a virtual step.

    The first parameter is either the mangled name of VTT global variable
    or the address of the subobject's sub-VTT entry in the VTT.
    The second parameter `offset` provides a virtual step to adjust to
    the actual address point of the vtable.

    The return type is always a `!cir.ptr<!cir.ptr<void>>`.

    Example:
    ```mlir
    cir.global linkonce_odr @_ZTV1B = ...
    ...
    %3 = cir.base_class_addr(%1 : !cir.ptr<!rec_D> nonnull) [0]
             -> !cir.ptr<!rec_B>
    %4 = cir.vtt.address_point @_ZTT1D, offset = 1
             -> !cir.ptr<!cir.ptr<!void>>
    cir.call @_ZN1BC2Ev(%3, %4)
    ```
    Or:
    ```mlir
    %7 = cir.vtt.address_point %3 : !cir.ptr<!cir.ptr<!void>>, offset = 1
             -> !cir.ptr<!cir.ptr<!void>>
    ```
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$name,
                       Optional<CIR_AnyType>:$sym_addr,
                       I32Attr:$offset);
  let results = (outs CIR_PointerType:$addr);

  let assemblyFormat = [{
      ($name^)?
      ($sym_addr^ `:` type($sym_addr))?
      `,`
      `offset` `=` $offset
    `->` qualified(type($addr)) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SetBitfieldOp
//===----------------------------------------------------------------------===//

def CIR_SetBitfieldOp : CIR_Op<"set_bitfield"> {
  let summary = "Set the value of a bitfield member";
  let description = [{
    The `cir.set_bitfield` operation provides a store-like access to
    a bit field of a record.

    A bitfield info attribute must be provided to describe the location of
    the bitfield within the memory referenced by the $addr argument.
    The $src argument is inserted at the appropriate place in the memory and
    the value that was stored. Returns the value being stored.

    A unit attribute `volatile` can be used to indicate a volatile store of the
    bitfield.
      ```mlir
        cir.set_bitfield(#bfi, %0 : !cir.ptr<!u32i>, %1 : !s32i) {is_volatile}
                                                                       -> !s32i
      ```

    Example.
    Suppose we have a struct with multiple bitfields stored in
    different storages. The `cir.set_bitfield` operation sets the value
    of the bitfield.
    ```C++
    typedef struct {
      int a : 4;
      int b : 27;
      int c : 17;
      int d : 2;
      int e : 15;
    } S;

    void store_bitfield(S& s) {
      s.e = 3;
    }
    ```

    ```mlir
    // 'e' is in the storage with the index 1
    !record_type = !cir.record<struct "S" packed padded {!u64i, !u16i,
                               !cir.array<!u8i x 2>} #cir.record.decl.ast>
    #bfi_e = #cir.bitfield_info<name = "e", storage_type = !u16i, size = 15,
                                offset = 0, is_signed = true>

    %1 = cir.const #cir.int<3> : !s32i
    %2 = cir.load %0 : !cir.ptr<!cir.ptr<!record_type>>, !cir.ptr<!record_type>
    %3 = cir.get_member %2[1] {name = "e"} : !cir.ptr<!record_type>
                                                             -> !cir.ptr<!u16i>
    %4 = cir.set_bitfield align(4) (#bfi_e, %3 : !cir.ptr<!u16i>, %1 : !s32i)
                                                                       -> !s32i
    ```
   }];

  let arguments = (ins
    Arg<CIR_PointerType, "the address to store the value", [MemWrite]>:$addr,
    CIR_AnyType:$src,
    CIR_BitfieldInfoAttr:$bitfield_info,
    DefaultValuedOptionalAttr<I64Attr, "0">:$alignment,
    UnitAttr:$is_volatile
  );

  let results = (outs CIR_IntType:$result);

  let assemblyFormat = [{
    (`align` `(` $alignment^ `)`)?
    `(`$bitfield_info`,` $addr`:`qualified(type($addr))`,`
    $src`:`type($src) `)`  attr-dict `->` type($result) }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$type,
                   "mlir::Value":$addr,
                   "mlir::Type":$storage_type,
                   "mlir::Value":$src,
                   "llvm::StringRef":$name,
                   "unsigned":$size,
                   "unsigned":$offset,
                   "bool":$is_signed,
                   "bool":$is_volatile,
                   CArg<"unsigned", "0">:$alignment
                   ),
   [{
      BitfieldInfoAttr info =
        BitfieldInfoAttr::get($_builder.getContext(),
                              name, storage_type,
                              size, offset, is_signed);
      build($_builder, $_state, type, addr, src, info, alignment, is_volatile);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// GetBitfieldOp
//===----------------------------------------------------------------------===//

def CIR_GetBitfieldOp : CIR_Op<"get_bitfield"> {
  let summary = "Get the information for a bitfield member";
  let description = [{
    The `cir.get_bitfield` operation provides a load-like access to
    a bit field of a record.

    It expects a name if a bit field, a pointer to a storage in the
    base record, a type of the storage, a name of the bitfield,
    a size the bit field, an offset of the bit field and a sign.

    A unit attribute `volatile` can be used to indicate a volatile load of the
    bitfield.
    ```mlir
      cir.get_bitfield(#bfi, %0 {is_volatile} : !cir.ptr<!u64i>) -> !s32i
    ```

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
    %4 = cir.get_bitfield align(4) (#bfi_e, %3 : !cir.ptr<!u16i>) -> !s32i
    ```
    }];

  let arguments = (ins
    Arg<CIR_PointerType, "the address to load from", [MemRead]>:$addr,
    CIR_BitfieldInfoAttr:$bitfield_info,
    DefaultValuedOptionalAttr<I64Attr, "0">:$alignment,
    UnitAttr:$is_volatile
    );

  let results = (outs CIR_IntType:$result);

  let assemblyFormat = [{
    (`align` `(` $alignment^ `)`)?
    `(`$bitfield_info `,` $addr attr-dict `:`
    qualified(type($addr)) `)` `->` type($result) }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$type,
                   "mlir::Value":$addr,
                   "mlir::Type":$storage_type,
                   "llvm::StringRef":$name,
                   "unsigned":$size,
                   "unsigned":$offset,
                   "bool":$is_signed,
                   "bool":$is_volatile,
                   CArg<"unsigned", "0">:$alignment
                   ),
   [{
      BitfieldInfoAttr info =
        BitfieldInfoAttr::get($_builder.getContext(),
                              name, storage_type,
                              size, offset, is_signed);
      build($_builder, $_state, type, addr, info, alignment, is_volatile);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// GetMemberOp
//===----------------------------------------------------------------------===//

def CIR_GetMemberOp : CIR_Op<"get_member"> {
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
// ExtractMemberOp
//===----------------------------------------------------------------------===//

def CIR_ExtractMemberOp : CIR_Op<"extract_member", [Pure]> {
  let summary = "Extract the value of a member of a record value";
  let description = [{
    The `cir.extract_member` operation extracts the value of a particular member
    from the input record. Unlike `cir.get_member` which derives pointers, this
    operation operates on values. It takes a value of record type and extracts
    the value of the specified record member from the input record value.

    Currently `cir.extract_member` does not work on unions.

    Example:

    ```mlir
    // Suppose we have a record with multiple members.
    !s32i = !cir.int<s, 32>
    !s8i = !cir.int<s, 32>
    !record_ty = !cir.record<"struct.Bar" {!s32i, !s8i}>

    // And suppose we have a value of the record type.
    %0 = cir.const #cir.const_record<{#cir.int<1> : !s32i,
	                                 #cir.int<2> : !s8i}> : !record_ty

    // Extract the value of the second member of the record.
    %1 = cir.extract_member %0[1] : !record_ty -> !s8i
    ```
  }];

  let arguments = (ins CIRRecordType:$record, I64Attr:$index);
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    $record `[` $index `]` attr-dict
    `:` qualified(type($record)) `->` qualified(type($result))
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$record, "uint64_t":$index), [{
      auto recordTy = mlir::cast<cir::RecordType>(record.getType());
      mlir::Type memberTy = recordTy.getMembers()[index];
      build($_builder, $_state, memberTy, record, index);
    }]>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

def CIR_GetElementOp : CIR_Op<"get_element", [
  TypesMatchWith<
      "type of 'result' matches element type of 'base'", "base", "result",
      "cir::PointerType::get(mlir::cast<cir::ArrayType>(mlir::cast<cir::"
      "PointerType>($_self).getPointee()).getElementType())">
]> {
  let summary = "Get the address of an array element";

  let description = [{
    The `cir.get_element` operation gets the address of a particular element
    from the `base` array.

    It expects a pointer to the `base` array and the `index` of the element.

    Example:
    ```mlir
    // Suppose we have a array.
    !s32i = !cir.int<s, 32>
    !arr_ty = !cir.array<!s32i x 4>

    // Get the address of the element at index 1.
    %elem_1 = cir.get_element %0[1 : !s32i] : !cir.ptr<!array_ty> -> !cir.ptr<!s32i>

    // Get the address of the element at index %i.
    %i = ...
    %elem_i = cir.get_element %0[%i : !s32i] : !cir.ptr<!array_ty> -> !cir.ptr<!s32i>

    ```
  }];

  let arguments = (ins
    Arg<CIR_PtrToArray, "the base address of the array ">:$base,
    Arg<CIR_AnyFundamentalIntType, "the index of the element">:$index
  );
  let results = (outs CIR_PointerType : $result);

  let assemblyFormat = [{
    $base`[` $index `:` type($index) `]` attr-dict
      `:` qualified(type($base)) `->` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    // Get the type of the element of the array.
    mlir::Type getElementType() {
      return getType().getPointee();
}

    cir::PointerType getBaseType() {
      return mlir::cast<cir::PointerType>(getBase().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// TODO(CIR): FuncOp is still a tiny shell of what it will become.  Many more
// properties and attributes will be added as upstreaming continues.

def CIR_OptionalPriorityAttr : OptionalAttr<
  DefaultValuedAttr<
    ConfinedAttr<I32Attr, [IntMinValue<101>, IntMaxValue<65535>]>,
    "65535"
  >
>;

// TODO(CIR): CallingConv is a placeholder here so we can use it in
// infrastructure calls, but it currently has no values.
def CIR_CallingConv : CIR_I32EnumAttr<"CallingConv", "calling convention", []>;

def CIR_FuncOp : CIR_Op<"func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  DeclareOpInterfaceMethods<CIRGlobalValueInterface>,
  HasAtMostOneOfAttrs<["global_ctor_priority", "global_dtor_priority"]>,
  IsolatedFromAbove
]> {
  let summary = "Declare or define a function";
  let description = [{
    The `cir.func` operation defines a function, similar to the `mlir::FuncOp`
    built-in.

    The function linkage information is specified by `linkage`, as defined by
    `GlobalLinkageKind` attribute.

    A compiler builtin function must be marked as `builtin` for further
    processing when lowering from CIR.

    The `coroutine` keyword is used to mark a coroutine function, which requires
    at least one `cir.await` instruction to be used in its body.

    The `lambda` translates to a C++ `operator()` that implements a lambda, this
    allow callsites to make certain assumptions about the real function nature
    when writing analysis.

    The `no_proto` keyword is used to identify functions that were declared
    without a prototype and, consequently, may contain calls with invalid
    arguments and undefined behavior.

    The `global_ctor` keyword indicates whether a function should execute before
    `main()` function, as specified by `__attribute__((constructor))`. An
    execution priority can also be specified `global_ctor(<priority>)`.
    Similarly, for global destructors both `global_dtor` and
    `global_dtor(<priority>)` are available.

    The `no_inline` attribute marks a function that should not be inlined.
    The `always_inline` attribute marks a function that should always be inlined.
    The `inline_hint` attribute suggests that the function should be inlined.

    The `personality` attribute specifies the personality function to use for
    exception handling. This is a symbol reference to the personality function
    (e.g., `@__gxx_personality_v0` for C++ exceptions).

    Example:

    ```mlir
    // External function definitions.
    cir.func @abort()

    // A function with internal linkage.
    cir.func internal @count(%x: i64) -> (i64)
      return %x : i64

    // Linkage information
    cir.func linkonce_odr @some_method(...)

    // Inline information
    cir.func no_inline @some_method(...)
    
    // Builtin function
    cir.func builtin @__builtin_coro_end(!cir.ptr<i8>, !cir.bool) -> !cir.bool
    // Coroutine
    cir.func coroutine @_Z10silly_taskv() -> !CoroTask {
      ...
      cir.await(...)
      ...
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    CIR_VisibilityAttr:$global_visibility,
    TypeAttrOf<CIR_FuncType>:$function_type,
    UnitAttr:$builtin,
    UnitAttr:$coroutine,
    OptionalAttr<CIR_InlineKind>:$inline_kind,
    UnitAttr:$lambda,
    UnitAttr:$no_proto,
    UnitAttr:$dso_local,
    DefaultValuedAttr<
      CIR_GlobalLinkageKind,
      "cir::GlobalLinkageKind::ExternalLinkage"
    >:$linkage,
    OptionalAttr<StrAttr>:$sym_visibility,
    UnitAttr:$comdat,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<FlatSymbolRefAttr>:$aliasee,
    OptionalAttr<CIR_SideEffect>:$side_effect,
    OptionalAttr<FlatSymbolRefAttr>:$personality,
    CIR_OptionalPriorityAttr:$global_ctor_priority,
    CIR_OptionalPriorityAttr:$global_dtor_priority,
    OptionalAttr<CIR_CXXSpecialMemberAttr>:$cxx_special_member
  );

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

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration();

    //===------------------------------------------------------------------===//
    // C++ Special Member Functions
    //===------------------------------------------------------------------===//

    /// Returns true if this function is a C++ special member function.
    bool isCXXSpecialMemberFunction();

    bool isCxxConstructor();
    bool isCxxDestructor();

    /// Returns true if this function is a copy or move assignment operator.
    bool isCxxSpecialAssignment();

    /// Returns the kind of constructor this function represents, if any.
    std::optional<CtorKind> getCxxConstructorKind();

    /// Returns the kind of assignment operator (move, copy) this function
    /// represents, if any.
    std::optional<AssignKind> getCxxSpecialAssignKind();

    /// Returns true if the function is a trivial C++ member functions such as
    /// trivial default constructor, copy/move constructor, copy/move assignment,
    /// or destructor.
    bool isCxxTrivialMemberFunction();
}];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let hasCXXABILowering = true;

  let extraLLVMLoweringPatternDecl = [{
    static mlir::StringRef getLinkageAttrNameString() { return "linkage"; }

    void lowerFuncAttributes(
        cir::FuncOp func, bool filterArgAndResAttrs,
        mlir::SmallVectorImpl<mlir::NamedAttribute> &result) const;

    mlir::LogicalResult
    matchAndRewriteAlias(cir::FuncOp op, llvm::StringRef aliasee, mlir::Type ty,
                         OpAdaptor adaptor,
                         mlir::ConversionPatternRewriter &rewriter) const;
  }];
}

//===----------------------------------------------------------------------===//
// LLVMIntrinsicCallOp
//===----------------------------------------------------------------------===//

def CIR_LLVMIntrinsicCallOp : CIR_Op<"call_llvm_intrinsic"> {
  let summary = "Call to llvm intrinsic functions that is not defined in CIR";
  let description = [{
    `cir.call_llvm_intrinsic` operation represents a call-like expression which has
    return type and arguments that maps directly to a llvm intrinsic.
    It only records intrinsic `intrinsic_name`.
  }];

  let results = (outs Optional<CIR_AnyType>:$result);
  let arguments = (ins
                   StrAttr:$intrinsic_name, Variadic<CIR_AnyType>:$arg_ops);

  let skipDefaultBuilders = 1;

  let assemblyFormat = [{
    $intrinsic_name $arg_ops `:` functional-type($arg_ops, $result) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$intrinsic_name, "mlir::Type":$resType,
              CArg<"mlir::ValueRange", "{}">:$operands), [{
      $_state.addAttribute("intrinsic_name", intrinsic_name);
      $_state.addOperands(operands);
      if (resType)
        $_state.addTypes(resType);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// CallOp and TryCallOp
//===----------------------------------------------------------------------===//

class CIR_CallOpBase<string mnemonic, list<Trait> extra_traits = []>
    : CIR_Op<mnemonic, !listconcat(extra_traits, [
        DeclareOpInterfaceMethods<CIRCallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
      ])> {
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

def CIR_CallOp : CIR_CallOpBase<"call", [NoRegionArguments]> {
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

  let skipDefaultBuilders = 1;

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

def CIR_TryCallOp : CIR_CallOpBase<"try_call",[
  Terminator
]> {
  let summary = "try_call operation";
  let description = [{
    Similar to `cir.call` but requires two destination blocks,
    one which is used if the call returns without throwing an
    exception (the "normal" destination) and another which is used
    if an exception is thrown (the "unwind" destination). 

    This operation is used only after the CFG flatterning pass.

    Example:

    ```mlir
    // Before CFG flattening
    cir.try {
      %call = cir.call @division(%a, %b) : () -> !s32i
      cir.yield
    } catch all {
      cir.yield
    }

    // After CFG flattening
    %call = cir.try_call @division(%a, %b) ^normalDest, ^unwindDest
      : (f32, f32) -> f32
    ^normalDest:
      cir.br ^afterTryBlock
    ^unwindDest:
      %exception_ptr, %type_id = cir.eh.inflight_exception
      cir.br ^catchHandlerBlock(%exception_ptr : !cir.ptr<!void>)
    ^catchHandlerBlock:
      ...
    ```
  }];

  let arguments = commonArgs;
  let results = (outs Optional<CIR_AnyType>:$result);
  let successors = (successor 
    AnySuccessor:$normalDest,
    AnySuccessor:$unwindDest
  );

  let skipDefaultBuilders = 1;
  let hasLLVMLowering = false;

  let builders = [
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee,
                "mlir::Type":$resType,
               "mlir::Block *":$normalDest,
               "mlir::Block *":$unwindDest,
               CArg<"mlir::ValueRange", "{}">:$callOperands,
               CArg<"SideEffect", "SideEffect::All">:$sideEffect), [{
      $_state.addOperands(callOperands);

      if (callee)
        $_state.addAttribute("callee", callee);
      if (resType && !isa<VoidType>(resType))
        $_state.addTypes(resType);

      $_state.addAttribute("side_effect",
        SideEffectAttr::get($_builder.getContext(), sideEffect));

      // Handle branches
      $_state.addSuccessors(normalDest);
      $_state.addSuccessors(unwindDest);
    }]>,
    OpBuilder<(ins "mlir::Value":$ind_target,
               "FuncType":$fn_type,
               "mlir::Block *":$normalDest,
               "mlir::Block *":$unwindDest,
               CArg<"mlir::ValueRange", "{}">:$callOperands,
               CArg<"SideEffect", "SideEffect::All">:$sideEffect), [{
      ::llvm::SmallVector<mlir::Value, 4> finalCallOperands({ind_target});
      finalCallOperands.append(callOperands.begin(), callOperands.end());
      $_state.addOperands(finalCallOperands);

      if (!fn_type.hasVoidReturn())
        $_state.addTypes(fn_type.getReturnType());

      $_state.addAttribute("side_effect",
        SideEffectAttr::get($_builder.getContext(), sideEffect));

      // Handle branches
      $_state.addSuccessors(normalDest);
      $_state.addSuccessors(unwindDest);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

def CIR_AwaitKind : CIR_I32EnumAttr<"AwaitKind", "await kind", [
  I32EnumAttrCase<"Init", 0, "init">,
  I32EnumAttrCase<"User", 1, "user">,
  I32EnumAttrCase<"Yield", 2, "yield">,
  I32EnumAttrCase<"Final", 3, "final">
]>;

def CIR_AwaitOp : CIR_Op<"await",[
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  RecursivelySpeculatable, NoRegionArguments
]> {
  let summary = "Wraps C++ co_await implicit logic";
  let description = [{
    The under the hood effect of using C++ `co_await expr` roughly
    translates to:

    ```c++
    // co_await expr;

    auto &&x = CommonExpr();
    if (!x.await_ready()) {
       ...
       x.await_suspend(...);
       ...
    }
    x.await_resume();
    ```

    `cir.await` represents this logic by using 3 regions:
      - ready: covers veto power from x.await_ready()
      - suspend: wraps actual x.await_suspend() logic
      - resume: handles x.await_resume()

    Breaking this up in regions allows individual scrutiny of conditions
    which might lead to folding some of them out. Lowerings coming out
    of CIR, e.g. LLVM, should use the `suspend` region to track more
    lower level codegen (e.g. intrinsic emission for coro.save/coro.suspend).

    There are also 4 flavors of `cir.await` available:
    - `init`: compiler generated initial suspend via implicit `co_await`.
    - `user`: also known as normal, representing a user written `co_await`.
    - `yield`: user written `co_yield` expressions.
    - `final`: compiler generated final suspend via implicit `co_await`.

    ```mlir
      cir.scope {
        ... // auto &&x = CommonExpr();
        cir.await(user, ready : {
          ... // x.await_ready()
        }, suspend : {
          ... // x.await_suspend()
        }, resume : {
          ... // x.await_resume()
        })
      }
    ```

    Note that resulution of the common expression is assumed to happen
    as part of the enclosing await scope.
  }];

  let arguments = (ins CIR_AwaitKind:$kind);
  let regions = (region SizedRegion<1>:$ready,
                        SizedRegion<1>:$suspend,
                        SizedRegion<1>:$resume);
  let assemblyFormat = [{
    `(` $kind `,`
    `ready` `:` $ready `,`
    `suspend` `:` $suspend `,`
    `resume` `:` $resume `,`
    `)`
    attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "cir::AwaitKind":$kind,
      CArg<"BuilderCallbackRef",
           "nullptr">:$readyBuilder,
      CArg<"BuilderCallbackRef",
           "nullptr">:$suspendBuilder,
      CArg<"BuilderCallbackRef",
           "nullptr">:$resumeBuilder
      )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def CIR_CopyOp : CIR_Op<"copy",[
  SameTypeOperands,
  DeclareOpInterfaceMethods<PromotableMemOpInterface>
]> {
  let summary = "Copies contents from a CIR pointer to another";
  let description = [{
    Given two CIR pointers, `src` and `dst`, `cir.copy` will copy the memory
    pointed by `src` to the memory pointed by `dst`.

    The number of bytes copied is inferred from the pointee type. The pointee
    type of `src` and `dst` must match and both must implement the
    `DataLayoutTypeInterface`.

    The `volatile` keyword indicates that the operation is volatile.

    Examples:

    ```mlir
      // Copying contents from one record to another:
      cir.copy %0 to %1 : !cir.ptr<!record_ty>
    ```
  }];

  let arguments = (ins
      Arg<CIR_PointerType, "", [MemWrite]>:$dst,
      Arg<CIR_PointerType, "", [MemRead]>:$src,
      UnitAttr:$is_volatile
  );

  let assemblyFormat = [{$src `to` $dst (`volatile` $is_volatile^)?
                        attr-dict `:` qualified(type($dst))
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the pointer type being copied.
    cir::PointerType getType() { return getSrc().getType(); }

    /// Returns the number of bytes to be copied.
    unsigned getLength(const mlir::DataLayout &dt) {
      return dt.getTypeSize(getType().getPointee());
    }    
  }];
}

//===----------------------------------------------------------------------===//
// MemCpyOp && MemMoveOp
//===----------------------------------------------------------------------===//

class CIR_MemOp<string mnemonic> : CIR_Op<mnemonic, [
  AllTypesMatch<["dst", "src"]>
]> {
  dag commonArgs = (ins
    Arg<CIR_VoidPtrType, "", [MemWrite]>:$dst,
    Arg<CIR_VoidPtrType, "", [MemRead]>:$src
  );
}

def CIR_MemCpyOp : CIR_MemOp<"libc.memcpy"> {
  let summary = "Equivalent to libc's `memcpy`";
  let description = [{
    Given two CIR pointers, `src` and `dst`, `cir.libc.memcpy` will copy `len`
    bytes from the memory pointed by `src` to the memory pointed by `dst`.

    While `cir.copy` is meant to be used for implicit copies in the code where
    the length of the copy is known, `cir.memcpy` copies only from and to void
    pointers, requiring the copy length to be passed as an argument.

    As is the case for memcpy in the C standard library, this operation
    exhibits undefined behavior (UB) if any of the following conditions hold:
      * `src` and/or `dst` are null pointers; or
      * the memory regions referenced by `src` and `dst` overlap.

    Examples:

    ```mlir
      // Copying 2 bytes from one array to a record:
      %2 = cir.const #cir.int<2> : !u32i
      cir.libc.memcpy %2 bytes from %arr to %record : !cir.ptr<!arr> -> !cir.ptr<!record>
    ```
  }];

  let arguments = !con(commonArgs, (ins CIR_AnyFundamentalUIntType:$len));

  let assemblyFormat = [{
    $len `bytes` `from` $src `to` $dst attr-dict
    `:` type($len) `,` qualified(type($src)) `->` qualified(type($dst))
  }];

  let extraClassDeclaration = [{
    /// Returns the byte length type.
    cir::IntType getLenTy() { return getLen().getType(); }
  }];
}

// TODO: MemMoveOp

//===----------------------------------------------------------------------===//
// ReturnAddrOp and FrameAddrOp
//===----------------------------------------------------------------------===//

def CIR_ReturnAddrOp : CIR_Op<"return_address"> {
  let summary =
      "The return address of the current function, or of one of its callers";

  let description = [{
    Represents a call to builtin function ` __builtin_return_address` in CIR.
    This builtin function returns the return address of the current function,
    or of one of its callers.

    The `level` argument is number of frames to scan up the call stack.
    For instance, value of 0 yields the return address of the current function,
    value of 1 yields the return address of the caller of the current function,
    and so forth.

    Examples:

    ```mlir
    %p = return_address(%level) : !cir.ptr<!void>
    ```
  }];

  let arguments = (ins CIR_UInt32:$level);
  let results = (outs CIR_VoidPtrType:$result);
  let assemblyFormat = "`(` $level `)` attr-dict";
}

def CIR_FrameAddrOp : CIR_Op<"frame_address"> {
  let summary =
      "The frame address of the current function, or of one of its callers";

  let description = [{
    Represents a call to builtin function ` __builtin_frame_address` in CIR.
    This builtin function returns the frame address of the current function,
    or of one of its callers. The frame is the area on the stack that holds
    local variables and saved registers. The frame address is normally the
    address of the first word pushed on to the stack by the function.
    However, the exact definition depends upon the processor and the calling
    convention. If the processor has a dedicated frame pointer register, and
    the function has a frame, then __builtin_frame_address returns the value of
    the frame pointer register.

    The `level` argument is number of frames to scan up the call stack.
    For instance, value of 0 yields the frame address of the current function,
    value of 1 yields the frame address of the caller of the current function,
    and so forth.

    Examples:

    ```mlir
    %p = frame_address(%level) : !cir.ptr<!u8i>
    ```
  }];

  let arguments = (ins CIR_UInt32:$level);
  let results = (outs CIR_PointerType:$result);
  let assemblyFormat = "`(` $level `)` attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// AddrOfReturnAddrOp
//===----------------------------------------------------------------------===//

def CIR_AddrOfReturnAddrOp : CIR_Op<"address_of_return_address"> {
  let summary = "The place stores the return address of the current function";

  let description = [{
    Represents a call to builtin function `_AddressOfReturnAddress` in CIR.
    This builtin function returns a pointer to the place in the stack frame
    where the return address of the current function is stored.

    Examples:

    ```mlir
    %addr = address_of_return_address() : !cir.ptr<!u8i>
    ```
  }];

  let results = (outs CIR_PointerType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// StackSaveOp & StackRestoreOp
//===----------------------------------------------------------------------===//

def CIR_StackSaveOp : CIR_Op<"stacksave"> {
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

def CIR_StackRestoreOp : CIR_Op<"stackrestore"> {
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
// InlineAsmOp
//===----------------------------------------------------------------------===//

def CIR_AsmFlavor : CIR_I32EnumAttr<"AsmFlavor", "ATT or Intel",
                                    [I32EnumAttrCase<"x86_att", 0>,
                                     I32EnumAttrCase<"x86_intel", 1>]>;

def CIR_InlineAsmOp : CIR_Op<"asm", [RecursiveMemoryEffects]> {
  let description = [{
    The `cir.asm` operation represents C/C++ asm inline.

    CIR constraints strings follow the same rules that are established for
    the C level assembler constraints with several differences caused by
    clang::AsmStmt processing.

    Thus, numbers that appears in the constraint string may also refer to:
    - the output variable index referenced by the input operands.
    - the index of early-clobber operand

    Operand attributes are a storage, where each element corresponds to the
    operand with the same index. The first index relates to the operation
    result (if any).
    The operands themselves are stored as VariadicOfVariadic in the following
    order: output, input and then in/out operands. When several output operands
    are present, the result type may be represented as an anonymous record type.

    Example:
    ```C++
    __asm__("foo" : : : );
    __asm__("bar $42 %[val]" : [val] "=r" (x), "+&r"(x));
    __asm__("baz $42 %[val]" : [val] "=r" (x), "+&r"(x) : "[val]"(y));
    ```

    ```mlir
    !rec_22anon2E022 = !cir.record<struct "anon.0" {!cir.int<s, 32>, !cir.int<s, 32>}>
    !rec_22anon2E122 = !cir.record<struct "anon.1" {!cir.int<s, 32>, !cir.int<s, 32>}>
    ...
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
    %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
    ...
    %2 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    %3 = cir.load %1 : !cir.ptr<!s32i>, !s32i

    cir.asm(x86_att,
      out = [],
      in = [],
      in_out = [],
      {"foo" "~{dirflag},~{fpsr},~{flags}"}) side_effects

    cir.asm(x86_att,
      out = [],
      in = [],
      in_out = [%2 : !s32i],
      {"bar $$42 $0" "=r,=&r,1,~{dirflag},~{fpsr},~{flags}"}) -> !rec_22anon2E022

    cir.asm(x86_att,
      out = [],
      in = [%3 : !s32i],
      in_out = [%2 : !s32i],
      {"baz $$42 $0" "=r,=&r,0,1,~{dirflag},~{fpsr},~{flags}"}) -> !rec_22anon2E122
    ```
  }];

  let results = (outs Optional<CIR_AnyType>:$res);

  let arguments =
      (ins VariadicOfVariadic<AnyType, "operands_segments">:$asm_operands,
          StrAttr:$asm_string, StrAttr:$constraints, UnitAttr:$side_effects,
          CIR_AsmFlavor:$asm_flavor, ArrayAttr:$operand_attrs,
          DenseI32ArrayAttr:$operands_segments);

  let builders = [OpBuilder<(ins
      "llvm::ArrayRef<mlir::ValueRange>":$asmOperands,
      "llvm::StringRef":$asmString, "llvm::StringRef":$constraints,
      "bool":$sideEffects, "AsmFlavor":$asmFlavor,
      "llvm::ArrayRef<mlir::Attribute>":$operandAttrs)>];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

def CIR_UnreachableOp : CIR_Op<"unreachable", [Terminator]> {
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

def CIR_TrapOp : CIR_Op<"trap", [Terminator]> {
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
// ArrayCtor & ArrayDtor
//===----------------------------------------------------------------------===//

class CIR_ArrayInitDestroy<string mnemonic> : CIR_Op<mnemonic> {
  let arguments = (ins
    Arg<CIR_PtrToArray, "array address", [MemWrite, MemRead]>:$addr
  );

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $addr `:` qualified(type($addr)) $body attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$addr,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$regionBuilder), [{
        assert(regionBuilder && "builder callback expected");
        mlir::OpBuilder::InsertionGuard guard($_builder);
        mlir::Region *r = $_state.addRegion();
        $_state.addOperands(ValueRange{addr});
        $_builder.createBlock(r);
        regionBuilder($_builder, $_state.location);
    }]>
  ];

  let hasLLVMLowering = false;
}

def CIR_ArrayCtor : CIR_ArrayInitDestroy<"array.ctor"> {
  let summary = "Initialize array elements with C++ constructors";
  let description = [{
    Initialize each array element using the same C++ constructor. This
    operation has one region, with one single block. The block has an
    incoming argument for the current array element to initialize.

    Example:

    ```mlir
    cir.array.ctor(%0 : !cir.ptr<!cir.array<!rec_S x 42>>) {
      ^bb0(%arg0: !cir.ptr<!rec_S>):
        cir.call @some_ctor(%arg0) : (!cir.ptr<!rec_S>) -> ()
        cir.yield
    }
    ```
  }];
}

def CIR_ArrayDtor : CIR_ArrayInitDestroy<"array.dtor"> {
  let summary = "Destroy array elements with C++ dtors";
  let description = [{
    Destroy each array element using the same C++ destructor. This
    operation has one region, with one single block. The block has an
    incoming argument for the current array element to destruct.

    Example:

    ```mlir
    cir.array.dtor(%0 : !cir.ptr<!cir.array<!rec_S x 42>>) {
      ^bb0(%arg0: !cir.ptr<!rec_S>):
        cir.call @some_dtor(%arg0) : (!cir.ptr<!rec_S>) -> ()
        cir.yield
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// GetRuntimeMemberOp
//===----------------------------------------------------------------------===//

def CIR_GetRuntimeMemberOp : CIR_Op<"get_runtime_member"> {
  let summary = "Get the address of a member of a record";
  let description = [{
    The `cir.get_runtime_member` operation gets the address of a member from
    the input record. The target member is given by a value of type
    `!cir.data_member` (i.e. a pointer-to-data-member value).

    This operation differs from `cir.get_member` in when the target member can
    be determined. For the `cir.get_member` operation, the target member is
    specified as a constant index so the member it returns access to is known
    when the operation is constructed. For the `cir.get_runtime_member`
    operation, the target member is given through a pointer-to-data-member
    value which is unknown until the program being compiled is executed. In
    other words, `cir.get_member` represents a normal member access through the
    `.` operator in C/C++:

    ```cpp
    struct Foo { int x; };
    Foo f;
    (void)f.x;  // cir.get_member
    ```

    And `cir.get_runtime_member` represents a member access through the `.*` or
    the `->*` operator in C++:

    ```cpp
    struct Foo { int x; }
    Foo f;
    Foo *p;
    int Foo::*member;

    (void)f.*member;   // cir.get_runtime_member
    (void)p->*member;  // cir.get_runtime_member
    ```

    This operation expects a pointer to the base record as well as the pointer
    to the target member.
  }];

  let arguments = (ins
    Arg<CIR_PtrToRecordType, "address of the record object", [MemRead]>:$addr,
    Arg<CIR_DataMemberType, "pointer to the target member">:$member);

  let results = (outs Res<CIR_PointerType, "">:$result);

  let assemblyFormat = [{
    $addr `[` $member `:` qualified(type($member)) `]` attr-dict
    `:` qualified(type($addr)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;

  let hasCXXABILowering = true;
  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// GetMethodOp
//===----------------------------------------------------------------------===//

def CIR_GetMethodOp : CIR_Op<"get_method"> {
  let summary = "Resolve a method to a function pointer as callee";
  let description = [{
    The `cir.get_method` operation takes a pointer to method (!cir.method) and
    a pointer to a class object (!cir.ptr<!cir.record>>) as input, and
    yields a function pointer that points to the actual function corresponding
    to the input method. The operation also applies any necessary adjustments to
    the input object pointer for calling the method and yields the adjusted
    pointer.

    This operation is generated when calling a method through a pointer-to-
    member-function in C++:

    ```cpp
    // Foo *object;
    // int arg;
    // void (Foo::*method)(int);

    (object->*method)(arg);
    ```

    The code above will generate CIR similar to:

    ```mlir
    %callee, %this = cir.get_method %method, %object
    cir.call %callee(%this, %arg)
    ```

    The method type must match the callee type. That is:
    - The return type of the method must match the return type of the callee.
    - The first parameter of the callee must have type `!cir.ptr<!cir.void>`.
    - Types of other parameters of the callee must match the parameters of the
      method.
  }];

  let arguments = (ins CIR_MethodType:$method, CIR_PtrToRecordType:$object);
  let results = (outs CIR_PtrToFunc:$callee, CIR_VoidPtrType:$adjusted_this);

  let assemblyFormat = [{
    $method `,` $object
    `:` `(` qualified(type($method)) `,` qualified(type($object)) `)`
    `->` `(` qualified(type($callee)) `,` qualified(type($adjusted_this)) `)`
    attr-dict
  }];

  let hasVerifier = 1;
  let hasLLVMLowering = false;
  let hasCXXABILowering = true;
}

//===----------------------------------------------------------------------===//
// VecCreate
//===----------------------------------------------------------------------===//

def CIR_VecCreateOp : CIR_Op<"vec.create", [Pure]> {
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

def CIR_VecInsertOp : CIR_Op<"vec.insert", [
  Pure,
  TypesMatchWith<"argument type matches vector element type",
    "vec", "value", "mlir::cast<cir::VectorType>($_self).getElementType()">,
  AllTypesMatch<["result", "vec"]>
]> {
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

def CIR_VecExtractOp : CIR_Op<"vec.extract", [
  Pure,
  TypesMatchWith<"type of 'result' matches element type of 'vec'",
    "vec", "result", "mlir::cast<cir::VectorType>($_self).getElementType()">
]> {
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

def CIR_VecCmpOp : CIR_Op<"vec.cmp", [Pure, SameTypeOperands]> {
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

def CIR_VecShuffleOp : CIR_Op<"vec.shuffle", [
  Pure, AllTypesMatch<["vec1", "vec2"]>
]> {
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

def CIR_VecShuffleDynamicOp : CIR_Op<"vec.shuffle.dynamic", [
  Pure, AllTypesMatch<["vec", "result"]>
]> {
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

def CIR_VecTernaryOp : CIR_Op<"vec.ternary", [
  Pure, AllTypesMatch<["result", "lhs", "rhs"]>
]> {
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

def CIR_VecSplatOp : CIR_Op<"vec.splat", [
  Pure,
  TypesMatchWith<"type of 'value' matches element type of 'result'",
    "result", "value", "mlir::cast<cir::VectorType>($_self).getElementType()">
]> {
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

def CIR_BaseClassAddrOp : CIR_Op<"base_class_addr"> {
  let summary = "Get the base class address for a class/struct";
  let description = [{
    The `cir.base_class_addr` operaration gets the address of a particular
    non-virtual base class given a derived class pointer. The offset in bytes
    of the base class must be passed in, since it is easier for the front end
    to calculate that than the MLIR passes. The operation contains a flag for
    whether or not the operand may be nullptr. That depends on the context and
    cannot be known by the operation, and that information affects how the
    operation is lowered.

    The validity of the relationship of derived and base cannot yet be verified.
    If the target class is not a valid base class for the object, the behavior
    is undefined.

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
// DerivedClassAddrOp
//===----------------------------------------------------------------------===//

def CIR_DerivedClassAddrOp : CIR_Op<"derived_class_addr"> {
  let summary = "Get the derived class address for a class/struct";
  let description = [{
    The `cir.derived_class_addr` operaration gets the address of a particular
    derived class given a non-virtual base class pointer. The offset in bytes
    of the base class must be passed in, similar to `cir.base_class_addr`, but
    going into the other direction. This means lowering to a negative offset.

    The operation contains a flag for whether or not the operand may be nullptr.
    That depends on the context and cannot be known by the operation, and that
    information affects how the operation is lowered.

    The validity of the relationship of derived and base cannot yet be verified.
    If the target class is not a valid derived class for the object, the
    behavior is undefined.

    Example:
    ```c++
    class A {};
    class B : public A {};

    B *getAsB(A *a) {
      return static_cast<B*>(a);
    }
    ```

    leads to
    ```mlir
      %2 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
      %3 = cir.base_class_addr %2 : !cir.ptr<!rec_B> [0] -> !cir.ptr<!rec_A>
    ```
  }];

  let arguments = (ins
    Arg<CIR_PointerType, "base class pointer", [MemRead]>:$base_addr,
    IndexAttr:$offset, UnitAttr:$assume_not_null);

  let results = (outs Res<CIR_PointerType, "">:$derived_addr);

  let assemblyFormat = [{
      $base_addr `:` qualified(type($base_addr))
      (`nonnull` $assume_not_null^)?
      ` ` `[` $offset `]` `->` qualified(type($derived_addr)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BaseDataMemberOp & DerivedDataMemberOp
//===----------------------------------------------------------------------===//

def CIR_BaseDataMemberOp : CIR_Op<"base_data_member", [Pure]> {
  let summary =
    "Cast a derived class data member pointer to a base class data member "
    "pointer";
  let description = [{
    The `cir.base_data_member` operation casts a data member pointer of type
    `T Derived::*` to a data member pointer of type `T Base::*`, where `Base`
    is an accessible non-ambiguous non-virtual base class of `Derived`.

    The `offset` parameter gives the offset in bytes of the `Base` base class
    subobject within a `Derived` object.
  }];

  let arguments = (ins CIR_DataMemberType:$src, IndexAttr:$offset);
  let results = (outs CIR_DataMemberType:$result);

  let assemblyFormat = [{
    $src `[` $offset `]` `:` qualified(type($src)) `->` qualified(type($result)) attr-dict
  }];

  let hasVerifier = 1;
  let hasLLVMLowering = false;
  let hasCXXABILowering = true;  
}

def CIR_DerivedDataMemberOp : CIR_Op<"derived_data_member", [Pure]> {
  let summary =
    "Cast a base class data member pointer to a derived class data member "
    "pointer";
  let description = [{
    The `cir.derived_data_member` operation casts a data member pointer of type
    `T Base::*` to a data member pointer of type `T Derived::*`, where `Base`
    is an accessible non-ambiguous non-virtual base class of `Derived`.

    The `offset` parameter gives the offset in bytes of the `Base` base class
    subobject within a `Derived` object.
  }];

  let arguments = (ins CIR_DataMemberType:$src, IndexAttr:$offset);
  let results = (outs CIR_DataMemberType:$result);

  let assemblyFormat = [{
    $src `[` $offset `]` `:` qualified(type($src)) `->` qualified(type($result)) attr-dict
  }];

  let hasVerifier = 1;
  let hasLLVMLowering = false;
  let hasCXXABILowering = true;  
}

//===----------------------------------------------------------------------===//
// ComplexCreateOp
//===----------------------------------------------------------------------===//

def CIR_ComplexCreateOp : CIR_Op<"complex.create", [Pure, SameTypeOperands]> {
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

def CIR_ComplexRealOp : CIR_Op<"complex.real", [Pure]> {
  let summary = "Extract the real part of a complex value";
  let description = [{
    `cir.complex.real` operation takes an operand of `!cir.complex`, `cir.int`, 
    `!cir.bool` or `!cir.float`. If the operand is `!cir.complex`, the real 
    part of it will be returned, otherwise the value returned unmodified. 

    Example:

    ```mlir
    %real = cir.complex.real %complex : !cir.complex<!cir.float> -> !cir.float
    %real = cir.complex.real %scalar : !cir.float -> !cir.float
    ```
  }];

  let results = (outs CIR_AnyIntOrBoolOrFloatType:$result);
  let arguments = (ins CIR_AnyComplexOrIntOrBoolOrFloatType:$operand);

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

def CIR_ComplexImagOp : CIR_Op<"complex.imag", [Pure]> {
  let summary = "Extract the imaginary part of a complex value";
  let description = [{
    `cir.complex.imag` operation takes an operand of `!cir.complex`, `!cir.int`
    `!cir.bool` or `!cir.float`. If the operand is `!cir.complex`, the imag 
    part of it will be returned, otherwise a zero value will be returned.  

    Example:

    ```mlir
    %imag = cir.complex.imag %complex : !cir.complex<!cir.float> -> !cir.float
    %imag = cir.complex.imag %scalar : !cir.float -> !cir.float
    ```
  }];

  let results = (outs CIR_AnyIntOrBoolOrFloatType:$result);
  let arguments = (ins CIR_AnyComplexOrIntOrBoolOrFloatType:$operand);

  let assemblyFormat = [{
    $operand `:` qualified(type($operand)) `->` qualified(type($result))
    attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ComplexRealPtrOp
//===----------------------------------------------------------------------===//

def CIR_ComplexRealPtrOp : CIR_Op<"complex.real_ptr", [Pure]> {
  let summary = "Derive a pointer to the real part of a complex value";
  let description = [{
    `cir.complex.real_ptr` operation takes a pointer operand that points to a
    complex value of type `!cir.complex` and yields a pointer to the real part
    of the operand.

    Example:

    ```mlir
    %1 = cir.complex.real_ptr %0 : !cir.ptr<!cir.complex<!cir.double>>
      -> !cir.ptr<!cir.double>
    ```
  }];

  let results = (outs CIR_PtrToIntOrFloatType:$result);
  let arguments = (ins CIR_PtrToComplexType:$operand);

  let assemblyFormat = [{
    $operand `:`
    qualified(type($operand)) `->` qualified(type($result)) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ComplexImagPtrOp
//===----------------------------------------------------------------------===//

def CIR_ComplexImagPtrOp : CIR_Op<"complex.imag_ptr", [Pure]> {
  let summary = "Derive a pointer to the imaginary part of a complex value";
  let description = [{
    `cir.complex.imag_ptr` operation takes a pointer operand that points to a
    complex value of type `!cir.complex` and yields a pointer to the imaginary
    part of the operand.

    Example:

    ```mlir
    %1 = cir.complex.imag_ptr %0 : !cir.ptr<!cir.complex<!cir.double>>
      -> !cir.ptr<!cir.double>
    ```
  }];

  let arguments = (ins CIR_PtrToComplexType:$operand);
  let results = (outs CIR_PtrToIntOrFloatType:$result);

  let assemblyFormat = [{
    $operand `:`
    qualified(type($operand)) `->` qualified(type($result)) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ComplexAddOp
//===----------------------------------------------------------------------===//

def CIR_ComplexAddOp : CIR_Op<"complex.add", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Complex addition";
  let description = [{
    The `cir.complex.add` operation takes two complex numbers and returns
    their sum.

    Example:

    ```mlir
    %2 = cir.complex.add %0, %1 : !cir.complex<!cir.float>
    ```
  }];

  let arguments = (ins CIR_ComplexType:$lhs, CIR_ComplexType:$rhs);

  let results = (outs CIR_ComplexType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `:` qualified(type($result)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ComplexSubOp
//===----------------------------------------------------------------------===//

def CIR_ComplexSubOp : CIR_Op<"complex.sub", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Complex subtraction";
  let description = [{
    The `cir.complex.sub` operation takes two complex numbers and returns
    their difference.

    Example:

    ```mlir
    %2 = cir.complex.sub %0, %1 : !cir.complex<!cir.float>
    ```
  }];

  let arguments = (ins CIR_ComplexType:$lhs, CIR_ComplexType:$rhs);

  let results = (outs CIR_ComplexType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `:` qualified(type($result)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ComplexMulOp & ComplexDivOp
//===----------------------------------------------------------------------===//

def CIR_ComplexRangeKind : CIR_I32EnumAttr<
  "ComplexRangeKind", "complex multiplication and division implementation", [
    I32EnumAttrCase<"Full", 0, "full">,
    I32EnumAttrCase<"Improved", 1, "improved">,
    I32EnumAttrCase<"Promoted", 2, "promoted">,
    I32EnumAttrCase<"Basic", 3, "basic">,
]>;

def CIR_ComplexMulOp : CIR_Op<"complex.mul", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Complex multiplication";
  let description = [{
    The `cir.complex.mul` operation takes two complex numbers and returns
    their product.

    For complex types with floating-point components, the `range` attribute
    specifies the algorithm to be used when the operation is lowered to
    the LLVM dialect. For multiplication, 'improved', 'promoted', and 'basic'
    are all handled equivalently, producing the algebraic formula with no
    special handling for NaN value. If 'full' is used, a runtime-library
    function is called if one of the intermediate calculations produced
    a NaN value.

    Example:

    ```mlir
    %2 = cir.complex.mul %0, %1 range(basic) : !cir.complex<!cir.float>
    %2 = cir.complex.mul %0, %1 range(full) : !cir.complex<!cir.float>
    ```
  }];

  let arguments = (ins
    CIR_ComplexType:$lhs,
    CIR_ComplexType:$rhs,
    CIR_ComplexRangeKind:$range
  );

  let results = (outs CIR_ComplexType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `range` `(` $range `)` `:` qualified(type($result)) attr-dict
  }];

  let hasLLVMLowering = false;
}

def CIR_ComplexDivOp : CIR_Op<"complex.div", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Complex division";
  let description = [{
    The `cir.complex.div` operation takes two complex numbers and returns
    their quotient.

    For complex types with floating-point components, the `range` attribute
    specifies the algorithm to be used when the operation is lowered to
    the LLVM dialect. For division, 'improved' produces Smith's algorithms for
    Complex division with no additional handling for NaN values. If 'promoted'
    is used, the values are promoted to a higher precision type, if possible,
    and the calculation is performed using the algebraic formula, with
    no additional handling for NaN values. We fall back on Smith's algorithm
    when the target doesn't support a higher precision type. If 'full' is used,
    a runtime-library function is called if one of the intermediate
    calculations produced a NaN value. and for 'basic' algebraic formula with
    no additional handling for the NaN value will be used. For integers types
    `range` attribute will be ignored.

    Example:

    ```mlir
    %2 = cir.complex.div %0, %1 range(basic) : !cir.complex<!cir.float>
    %2 = cir.complex.div %0, %1 range(full) : !cir.complex<!cir.float>
    ```
  }];

  let arguments = (ins
    CIR_ComplexType:$lhs,
    CIR_ComplexType:$rhs,
    CIR_ComplexRangeKind:$range
  );

  let results = (outs CIR_ComplexType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `range` `(` $range `)` `:` qualified(type($result)) attr-dict
  }];

  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// Bit Manipulation Operations
//===----------------------------------------------------------------------===//

class CIR_BitOpBase<string mnemonic, TypeConstraint operandTy>
    : CIR_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins operandTy:$input);
  let results = (outs operandTy:$result);

  let assemblyFormat = [{
    $input `:` type($result) attr-dict
  }];

  let hasFolder = 1;
}

class CIR_BitZeroCountOpBase<string mnemonic, TypeConstraint operandTy>
    : CIR_BitOpBase<mnemonic, operandTy> {
  let arguments = (ins operandTy:$input, UnitAttr:$poison_zero);

  let assemblyFormat = [{
    $input (`poison_zero` $poison_zero^)?
    `:` type($result) attr-dict
  }];
}

def CIR_BitClrsbOp : CIR_BitOpBase<"clrsb", CIR_SIntOfWidths<[32, 64]>> {
  let summary = "Get the number of leading redundant sign bits in the input";
  let description = [{
    Compute the number of leading redundant sign bits in the input integer.

    The input integer must be a signed integer. The most significant bit of the
    input integer is the sign bit. The `cir.clrsb` operation returns the number
    of consecutive bits following the sign bit that are identical to the sign
    bit.

    The bit width of the input integer must be either 32 or 64.

    Examples:

    ```mlir
    // %0 = 0b1101_1110_1010_1101_1011_1110_1110_1111
    %0 = cir.const #cir.int<3735928559> : !s32i
    // %1 will be 1 because there is 1 bit following the most significant bit
    // that is identical to it.
    %1 = cir.clrsb %0 : !s32i

    // %2 = 1, 0b0000_0000_0000_0000_0000_0000_0000_0001
    %2 = cir.const #cir.int<1> : !s32i
    // %3 will be 30 because there are 30 consecutive bits following the sign
    // bit that are identical to the sign bit.
    %3 = cir.clrsb %2 : !s32i
    ```
  }];
}

def CIR_BitClzOp : CIR_BitZeroCountOpBase<"clz",
  CIR_UIntOfWidths<[16, 32, 64]>
> {
  let summary = "Get the number of leading 0-bits in the input";
  let description = [{
    Compute the number of leading 0-bits in the input.

    The input integer must be an unsigned integer. The `cir.clz` operation
    returns the number of consecutive 0-bits at the most significant bit
    position in the input.

    If the `poison_zero` attribute is present, this operation will have
    undefined behavior if the input value is 0.

    Example:

    ```mlir
    // %0 = 0b0000_0000_0000_0000_0000_0000_0000_1000
    %0 = cir.const #cir.int<8> : !u32i
    // %1 will be 28
    %1 = cir.clz %0 poison_zero : !u32i
    ```
  }];
}

def CIR_BitCtzOp : CIR_BitZeroCountOpBase<"ctz",
  CIR_UIntOfWidths<[16, 32, 64]>
> {
  let summary = "Get the number of trailing 0-bits in the input";
  let description = [{
    Compute the number of trailing 0-bits in the input.

    The input integer must be an unsigned integer. The `cir.ctz` operation
    counts the number of consecutive 0-bits starting from the least significant
    bit.

    If the `poison_zero` attribute is present, this operation will have
    undefined behavior if the input value is 0.

    Example:

    ```mlir
    // %0 = 0b1000
    %0 = cir.const #cir.int<8> : !u32i
    // %1 will be 3
    %1 = cir.ctz %0 poison_zero : !u32i
    ```
  }];
}

def CIR_BitFfsOp : CIR_BitOpBase<"ffs", CIR_SIntOfWidths<[32, 64]>> {
  let summary = "Get the position of the least significant 1-bit in input";
  let description = [{
    Compute the 1-based position of the least significant 1-bit of the input.

    The input integer must be a signed integer. The `cir.ffs` operation returns
    one plus the index of the least significant 1-bit of the input signed
    integer. If the input integer is 0, `cir.ffs` yields 0.

    Example:

    ```mlir
    !s32i = !cir.int<s, 32>

    // %0 = 0x0010_1000
    %0 = cir.const #cir.int<40> : !s32i
    // #1 will be 4 since the 4th least significant bit is 1.
    %1 = cir.ffs %0 : !s32i
    ```
  }];
}

def CIR_BitParityOp : CIR_BitOpBase<"parity", CIR_UIntOfWidths<[32, 64]>> {
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
    %1 = cir.parity %0 : !u32i
    ```
  }];
}

def CIR_BitPopcountOp : CIR_BitOpBase<"popcount",
  CIR_UIntOfWidths<[16, 32, 64]>
> {
  let summary = "Get the number of 1-bits in input";
  let description = [{
    Compute the number of 1-bits in the input.

    The input must be an unsigned integer.

    Example:

    ```mlir
    // %0 = 0x0110_1000
    %0 = cir.const #cir.int<104> : !u32i
    // %1 will be 3 since there are 3 1-bits in %0
    %1 = cir.popcount %0 : !u32i
    ```
  }];
}

def CIR_BitReverseOp : CIR_BitOpBase<"bitreverse",
  CIR_UIntOfWidths<[8, 16, 32, 64]>
> {
  let summary = "Reverse the bit pattern of the operand integer";
  let description = [{
    The `cir.bitreverse` operation reverses the bits of the operand integer. Its
    only argument must be of unsigned integer types of width 8, 16, 32, or 64.

    Example:

    ```mlir
    %1 = cir.bitreverse %0: !u32i
    ```
  }];
}

def CIR_ByteSwapOp : CIR_BitOpBase<"byte_swap",
  CIR_UIntOfWidths<[16, 32, 64]>
> {
  let summary = "Reverse the bytes in the object representation of the operand";
  let description = [{
    The `cir.byte_swap` operation takes an integer as operand, reverse the bytes
    in the object representation of the operand integer, and returns the result.

    The operand integer must be an unsigned integer. Its widths must be either
    16, 32, or 64.

    Example:

    ```mlir
    // %0 = 0x12345678
    %0 = cir.const #cir.int<305419896> : !u32i

    // %1 should be 0x78563412
    %1 = cir.byte_swap %0 : !u32i
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RotateOp
//===----------------------------------------------------------------------===//

def CIR_RotateOp : CIR_Op<"rotate", [Pure, SameOperandsAndResultType]> {
  let summary = "Rotate the bits in the operand integer";
  let description = [{
    The `cir.rotate` rotates the bits in `input` by the given amount `amount`.
    The rotate direction is specified by the `left` and `right` keyword.

    `input` must be an unsigned integer and its width must be either 8, 16, 32,
    or 64. The types of `input`, `amount`, and the result must all match.

    Example:

    ```mlir
    %r = cir.rotate left %0, %1 : !u32i
    %r = cir.rotate right %0, %1 : !u32i
    ```
  }];

  let results = (outs CIR_IntType:$result);
  let arguments = (ins
    CIR_UIntOfWidths<[8, 16, 32, 64]>:$input,
    CIR_IntType:$amount,
    UnitAttr:$rotateLeft
  );

  let assemblyFormat = [{
    (`left` $rotateLeft^) : (`right`)?
    $input `,` $amount `:` type($result) attr-dict
  }];

  let extraClassDeclaration = [{
    bool isRotateLeft() { return getRotateLeft(); }
    bool isRotateRight() { return !getRotateLeft(); }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FPClass Test Flags
//===----------------------------------------------------------------------===//

def FPClassTestEnum : CIR_I32EnumAttr<"FPClassTest", "floating-point class test flags", [
  // Basic flags
  I32EnumAttrCase<"SignalingNaN", 1, "fcSNan">,
  I32EnumAttrCase<"QuietNaN", 2, "fcQNan">,
  I32EnumAttrCase<"NegativeInfinity", 4, "fcNegInf">,
  I32EnumAttrCase<"NegativeNormal", 8, "fcNegNormal">,
  I32EnumAttrCase<"NegativeSubnormal", 16, "fcNegSubnormal">,
  I32EnumAttrCase<"NegativeZero", 32, "fcNegZero">,
  I32EnumAttrCase<"PositiveZero", 64, "fcPosZero">,
  I32EnumAttrCase<"PositiveSubnormal", 128, "fcPosSubnormal">,
  I32EnumAttrCase<"PositiveNormal", 256, "fcPosNormal">,
  I32EnumAttrCase<"PositiveInfinity", 512, "fcPosInf">,

  // Composite flags
  I32EnumAttrCase<"Nan", 3, "fcNan">,                   // fcSNan | fcQNan
  I32EnumAttrCase<"Infinity", 516, "fcInf">,            // fcPosInf | fcNegInf
  I32EnumAttrCase<"Normal", 264, "fcNormal">,           // fcPosNormal | fcNegNormal
  I32EnumAttrCase<"Subnormal", 144, "fcSubnormal">,     // fcPosSubnormal | fcNegSubnormal
  I32EnumAttrCase<"Zero", 96, "fcZero">,                // fcPosZero | fcNegZero
  I32EnumAttrCase<"PositiveFinite", 448, "fcPosFinite">,// fcPosNormal | fcPosSubnormal | fcPosZero
  I32EnumAttrCase<"NegativeFinite", 56, "fcNegFinite">, // fcNegNormal | fcNegSubnormal | fcNegZero
  I32EnumAttrCase<"Finite", 504, "fcFinite">,           // fcPosFinite | fcNegFinite
  I32EnumAttrCase<"Positive", 960, "fcPositive">,       // fcPosFinite | fcPosInf
  I32EnumAttrCase<"Negative", 60, "fcNegative">,        // fcNegFinite | fcNegInf
  I32EnumAttrCase<"All", 1023, "fcAllFlags">,           // fcNan | fcInf | fcFinite
]> {
  let cppNamespace = "::cir";
}

def CIR_IsFPClassOp : CIR_Op<"is_fp_class"> {
  let summary = "Corresponding to the `__builtin_fpclassify` builtin function in clang";

  let description = [{
    The `cir.is_fp_class` operation takes a floating-point value as its first
    argument and a bitfield of flags as its second argument. The operation
    returns a boolean value indicating whether the floating-point value
    satisfies the given flags.

    The flags must be a compile time constant and the values are:

    | Bit # | floating-point class |
    | ----- | -------------------- |
    |  0    | Signaling NaN        |
    |  1    | Quiet NaN            |
    |  2    | Negative infinity    |
    |  3    | Negative normal      |
    |  4    | Negative subnormal   |
    |  5    | Negative zero        |
    |  6    | Positive zero        |
    |  7    | Positive subnormal   |
    |  8    | Positive normal      |
    |  9    | Positive infinity    |
  }];

  let arguments = (ins CIR_AnyFloatType:$src,
                       FPClassTestEnum:$flags);
  let results = (outs CIR_BoolType:$result);
  let assemblyFormat = [{
    $src `,` $flags `:` functional-type($src, $result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Assume Operations
//===----------------------------------------------------------------------===//

def CIR_AssumeOp : CIR_Op<"assume"> {
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

def CIR_AssumeAlignedOp : CIR_Op<"assume_aligned", [
  Pure, AllTypesMatch<["pointer", "result"]>
]> {
  let summary = "Tell the optimizer that a pointer is aligned";
  let description = [{
    The `cir.assume_aligned` operation takes two or three arguments. The first
    argument `pointer` gives the pointer value whose alignment is to be assumed,
    and the second argument `align` is an integer attribute that gives the
    assumed alignment.

    The `offset` argument is optional. If given, it represents misalignment
    offset. When it's present, this operation tells the optimizer that the
    pointer is always misaligned to the alignment by `offset` bytes, a.k.a. the
    pointer yielded by `(char *)pointer - offset` is aligned to the specified
    alignment. Note that the `offset` argument is an SSA value rather than an
    attribute, which means that you could pass a dynamically determined value
    as the mialignment offset.

    The result of this operation has the same value as the `pointer` argument,
    but it additionally carries any alignment information indicated by this
    operation.

    This operation corresponds to the `__builtin_assume_aligned` builtin
    function.

    Example:

    ```mlir
    // Assume that %0 is a CIR pointer value of type !cir.ptr<!s32i>
    %1 = cir.assume_aligned %0 alignment 16 : !cir.ptr<!s32i>

    // With a misalignment offset of 4 bytes:
    %2 = cir.const #cir.int<4> : !u64i
    %3 = cir.assume_aligned %0 alignment 16 [offset %2 : !u64i] : !cir.ptr<!s32i>
    ```
  }];

  let arguments = (ins CIR_PointerType:$pointer,
                       I64Attr:$alignment,
                       Optional<CIR_IntType>:$offset);
  let results = (outs CIR_PointerType:$result);

  let assemblyFormat = [{
    $pointer
    `alignment` $alignment
    (`[` `offset` $offset^ `:` type($offset) `]`)?
    `:` qualified(type($pointer)) attr-dict
  }];
}

def CIR_AssumeSepStorageOp : CIR_Op<"assume_separate_storage", [
  SameTypeOperands
]> {
  let summary =
      "Tell the optimizer that two pointers point to different allocations";
  let description = [{
    The `cir.assume_separate_storage` operation takes two pointers as arguments,
    and the operation tells the optimizer that these two pointers point to
    different allocations.

    This operation corresponds to the `__builtin_assume_separate_storage`
    builtin function.
  }];

  let arguments = (ins CIR_VoidPtrType:$ptr1, CIR_VoidPtrType:$ptr2);

  let assemblyFormat = [{
    $ptr1 `,` $ptr2 `:` qualified(type($ptr1)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Branch Probability Operations
//===----------------------------------------------------------------------===//

def CIR_ExpectOp : CIR_Op<"expect", [
  Pure, AllTypesMatch<["result", "val", "expected"]>
]> {
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

//===----------------------------------------------------------------------===//
// PrefetchOp
//===----------------------------------------------------------------------===//

def CIR_PrefetchOp : CIR_Op<"prefetch"> {
  let summary = "Prefetch operation";
  let description = [{
    The `cir.prefetch` operation is a hint to the code generator to insert a
    prefetch instruction if supported; otherwise, it is a noop. Prefetches
    have no effect on the behavior of the program but can change its
    performance characteristics.

    ```mlir
    cir.prefetch(%0 : !cir.ptr<!void>) locality(1) write
    ```

    $locality is a temporal locality specifier ranging from (0) - no locality,
    to (3) - extremely local, keep in cache. If $locality is not present, the
    default value is 3.
    
    $isWrite specifies whether the prefetch is for a 'read' or 'write'. If
    $isWrite is not specified, it means that prefetch is prepared for 'read'.
  }];

  let arguments = (ins CIR_VoidPtrType:$addr,
      DefaultValuedAttr<ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<3>]>,
                        "3">:$locality,
      UnitAttr:$isWrite);

  let assemblyFormat = [{
        (`write` $isWrite^) : (`read`)?
        `locality` `(` $locality `)`
        $addr `:` qualified(type($addr))
        attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// ObjSizeOp
//===----------------------------------------------------------------------===//

def CIR_ObjSizeOp : CIR_Op<"objsize", [Pure]> {
  let summary = "Implements the llvm.objsize builtin";
  let description = [{
    The `cir.objsize` operation is designed to provide information to the
    optimizer to determine whether a) an operation (like memcpy) will
    overflow a buffer that corresponds to an object, or b) that a runtime
    check for overflow isn’t necessary. An object in this context means an
    allocation of a specific class, structure, array, or other object.

    When the `min` attribute is present, the operation returns the minimum
    guaranteed accessible size. When absent (max mode), it returns the maximum
    possible object size. Corresponds to `llvm.objectsize`'s `min` argument.

    The `dynamic` attribute determines if the value should be evaluated at
    runtime. Corresponds to `llvm.objectsize`'s `dynamic` argument.

    The `nullunknown` attribute controls how null pointers are handled. When
    present, null pointers are treated as having unknown size. When absent,
    null pointers are treated as having 0 size (in min mode) or -1 size
    (in max mode). Corresponds to `llvm.objectsize`'s `nullunknown` argument.

    Example:

    ```mlir
    %size = cir.objsize min %ptr : !cir.ptr<i32> -> i64
    %dsize = cir.objsize max dynamic %ptr : !cir.ptr<i32> -> i64
    %nsize = cir.objsize min nullunknown %ptr : !cir.ptr<i32> -> i64
    ```
  }];

  let arguments = (ins
    CIR_PointerType:$ptr,
    UnitAttr:$min,
    UnitAttr:$nullunknown,
    UnitAttr:$dynamic
  );

  let results = (outs CIR_AnyFundamentalIntType:$result);

  let assemblyFormat = [{
      (`min` $min^) : (`max`)?
      (`nullunknown` $nullunknown^)?
      (`dynamic` $dynamic^)?
      $ptr `:` qualified(type($ptr)) `->` qualified(type($result)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// PtrDiffOp
//===----------------------------------------------------------------------===//

def CIR_PtrDiffOp : CIR_Op<"ptr_diff", [Pure, SameTypeOperands]> {
  let summary = "Pointer subtraction arithmetic";
  let description = [{
    The cir.ptr_diff operation computes the difference between two pointers that
    have the same element type.

    The result reflects the ABI-defined size of the pointed-to type. For example,
    subtracting two !cir.ptr<!u64i> values may yield 1, representing an 8-byte
    difference. In contrast, for pointers to void or function types, a result of
    8 corresponds to an 8-byte difference.

    For pointers to types whose size are not aligned with the target data
    layout, the size is generally rounded to the next power of 2 bits. For
    example, subtracting two !cir.ptr<!s24i> values for the _BitInt(24) type may
    yield 1, representing a 4-byte difference (as opposed to a 3-byte
    difference).

    Example:

    ```mlir
    %7 = cir.ptr_diff %0, %1 : !cir.ptr<!u64i> -> !u64i
    ```
  }];

  let arguments = (ins CIR_PointerType:$lhs, CIR_PointerType:$rhs);
  let results = (outs CIR_AnyFundamentalIntType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs  `:` qualified(type($lhs)) `->` qualified(type($result)) 
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Floating Point Ops
//===----------------------------------------------------------------------===//

class CIR_UnaryFPToFPBuiltinOp<string mnemonic, string llvmOpName>
    : CIR_Op<mnemonic, [Pure, SameOperandsAndResultType]>
{
  let arguments = (ins CIR_AnyFloatOrVecOfFloatType:$src);
  let results = (outs CIR_AnyFloatOrVecOfFloatType:$result);

  let assemblyFormat = "$src `:` type($src) attr-dict";

  let llvmOp = llvmOpName;
}

def CIR_SqrtOp : CIR_UnaryFPToFPBuiltinOp<"sqrt", "SqrtOp"> {
  let summary = "Floating-point square root operation";
  
  let description = [{
    Computes the square root of a floating-point value or vector.

    The input must be either:
      • a floating-point scalar type, or
      • a vector whose element type is floating-point.

    The result type must match the input type exactly.

    Examples:
      // scalar
      %r = cir.sqrt %x : !cir.fp64

      // vector
      %v = cir.sqrt %vec : !cir.vector<!cir.fp32 x 4>
  }];
}

def CIR_ACosOp : CIR_UnaryFPToFPBuiltinOp<"acos", "ACosOp"> {
  let summary = "Computes the arcus cosine of the specified value";
  let description = [{
    `cir.acos`computes the arcus cosine of a given value and
    returns a result of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.
  }];
}

def CIR_ASinOp : CIR_UnaryFPToFPBuiltinOp<"asin", "ASinOp"> {
  let summary = "Computes the arcus sine of the specified value";
  let description = [{
    `cir.asin`computes the arcus sine of a given value and
    returns a result of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.
  }];
}

def CIR_ATanOp : CIR_UnaryFPToFPBuiltinOp<"atan", "ATanOp"> {
  let summary = "Computes the floating-point arcus tangent value";
  let description = [{
    `cir.atan` computes the arcus tangent of a floating-point operand
    and returns a result of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.
  }];
}

def CIR_CeilOp : CIR_UnaryFPToFPBuiltinOp<"ceil", "FCeilOp"> {
  let summary = "Computes the ceiling of the specified value";
  let description = [{
    `cir.ceil` computes the ceiling of a given value and returns a result
    of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.
  }];
}

def CIR_CosOp : CIR_UnaryFPToFPBuiltinOp<"cos", "CosOp"> {
  let summary = "Computes the floating-point cosine value";
  let description = [{
    `cir.cos` computes the cosine of a floating-point operand and returns
    a result of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.
  }];
}

def CIR_ExpOp : CIR_UnaryFPToFPBuiltinOp<"exp", "ExpOp"> {
  let summary = "Computes the floating-point base-e exponential value";
  let description = [{
    `cir.exp` computes the exponential of a floating-point operand and returns
    a result of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.
  }];
}

def CIR_Exp2Op : CIR_UnaryFPToFPBuiltinOp<"exp2", "Exp2Op"> {
  let summary = "Computes the floating-point base-2 exponential value";
  let description = [{
    `cir.exp2` computes the base-2 exponential of a floating-point operand and
     returns a result of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.
  }];
}

def CIR_FAbsOp : CIR_UnaryFPToFPBuiltinOp<"fabs", "FAbsOp"> {
  let summary = "Computes the floating-point absolute value";
  let description = [{
    `cir.fabs` computes the absolute value of a floating-point operand
    and returns a result of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.
  }];
}

def CIR_FloorOp : CIR_UnaryFPToFPBuiltinOp<"floor", "FloorOp"> {
  let summary = "Computes the floating-point floor value";
  let description = [{
    `cir.floor` computes the floor of a floating-point operand and returns
    a result of the same type.

    Floating-point exceptions are ignored, and it does not set `errno`.

    Example:

    ```mlir
    // $x : !cir.double
    %y = cir.floor %x : !cir.double
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Variadic Operations
//===----------------------------------------------------------------------===//

def CIR_VAStartOp : CIR_Op<"va_start"> {
  let summary = "Starts a variable argument list";
  let description = [{
    The cir.va_start operation models the C/C++ va_start macro by
    initializing a variable argument list at the given va_list storage
    location.

    The first operand must be a pointer to the target's `va_list`
    representation. This operation has no results and produces its effect by
    mutating the storage referenced by the pointer operand. The second operand
    must be an integer value that contains the expected number of arguments in
    that list.

    Each `cir.va_start` must be paired with a corresponding `cir.va_end`
    on the same logical `va_list` object along all control-flow paths. After
    `cir.va_end`, the `va_list` must not be accessed unless reinitialized
    with another `cir.va_start`.

    Lowering maps this to the LLVM intrinsic `llvm.va_start`, passing the
    appropriately decayed pointer to the underlying `va_list` storage.

    Example:

    ```mlir
    // %args : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
    %p = cir.cast array_to_ptrdecay %args
          : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>)
          -> !cir.ptr<!rec___va_list_tag>
    %count = cir.load %0 : !cir.ptr<!s32i>, !s32i
    cir.va_start %p %count : !cir.ptr<!rec___va_list_tag>, !s32i
    ```
  }];
  let arguments = (ins
    CIR_PointerType:$arg_list,
    CIR_AnyFundamentalIntType:$count
  );

  let assemblyFormat = [{
    $arg_list $count attr-dict `:` type(operands)
  }];
}

def CIR_VAEndOp : CIR_Op<"va_end"> {
  let summary = "Ends a variable argument list";
  let description = [{
    The `cir.va_end` operation models the C/C++ va_end macro by finalizing
    and cleaning up a variable argument list previously initialized with
    `cir.va_start`.

    The operand must be a pointer to the target's `va_list` representation.
    This operation has no results and produces its effect by mutating the
    storage referenced by the pointer operand.

    `cir.va_end` must only be called after a matching `cir.va_start` on the
    same `va_list` along all control-flow paths. After `cir.va_end`, the
    `va_list` is invalid and must not be accessed unless reinitialized.

    Lowering typically maps this to the LLVM intrinsic `llvm.va_end`,
    passing the appropriately decayed pointer to the underlying `va_list`
    storage.

    Example:
    ```mlir
    // %args : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
    %p = cir.cast array_to_ptrdecay %args
          : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
          -> !cir.ptr<!rec___va_list_tag>
    cir.va_end %p : !cir.ptr<!rec___va_list_tag>
    ```
  }];

  let arguments = (ins CIR_PointerType:$arg_list);

  let assemblyFormat = [{
    $arg_list attr-dict `:` type(operands)
  }];
}

def CIR_VACopyOp : CIR_Op<"va_copy"> {
  let summary = "Copied a variable argument list";
  let description = [{
    The `cir.copy` operation models the C/C++ va_copy macro.
    The variable argument list passed as the `$src_list` is copied to an
    unitialized `va_list` in the destination operand. The next argument that
    can be extracted from the copied list is the same as the next argument in
    the source list. The copied list must be destroyed with `va_end`.

    Example:

    ```mlir
    // %args : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
    %p = cir.cast array_to_ptrdecay %args
          : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
          -> !cir.ptr<!rec___va_list_tag>
    cir.va_copy %p to %dst
          : (!cir.ptr<!rec___va_list_tag>, !cir.ptr<!rec___va_list_tag>)
    ```
  }];

  let arguments = (ins
    CIR_PointerType:$dst_list,
    CIR_PointerType:$src_list
  );

  let assemblyFormat = [{
    $src_list `to` $dst_list attr-dict `:` type(operands)
  }];
}

def CIR_VAArgOp : CIR_Op<"va_arg"> {
  let summary = "Fetches next variadic element as a given type";
  let description = [{
    The `cir.va_arg` operation models the C/C++ `va_arg` macro by reading the
    next argument from an active variable argument list and producing it as a
    value of a specified result type.

    The operand must be a pointer to the target's `va_list` representation.
    The operation advances the `va_list` state as a side effect and returns
    the fetched value as the result, whose type is chosen by the user of the
    operation.

    A `cir.va_arg` must only be used on a `va_list` that has been initialized
    with `cir.va.start` and not yet finalized by `cir.va.end`. The semantics
    (including alignment and promotion rules) follow the platform ABI; the
    frontend is responsible for providing a `va_list` pointer that matches the
    target representation.

    Example:
    ```mlir
    // %args : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
    %p = cir.cast array_to_ptrdecay %args
            : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
            -> !cir.ptr<!rec___va_list_tag>
    cir.va.start %p : !cir.ptr<!rec___va_list_tag>

    // Fetch an `int` from the vararg list.
    %v = cir.va_arg %p : (!cir.ptr<!rec___va_list_tag>) -> !s32i

    cir.va.end %p : !cir.ptr<!rec___va_list_tag>
    ```
  }];

  let arguments = (ins CIR_PointerType:$arg_list);
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    $arg_list attr-dict `:` functional-type(operands, $result)
  }];
}

//===----------------------------------------------------------------------===//
// ThrowOp
//===----------------------------------------------------------------------===//

def CIR_ThrowOp : CIR_Op<"throw"> {
  let summary = "(Re)Throws an exception";
  let description = [{
    This operation is equivalent to either __cxa_throw or __cxa_rethrow,
    depending on the arguments.

    The absense of arguments for `cir.throw` means it rethrows.

    For the no-rethrow version, it must have at least two operands, the RTTI
    information, a pointer to the exception object (likely allocated via
    `cir.alloc_exception`) and finally an optional dtor, which might run as
    part of this operation.

    Example:

    ```mlir
    // re-throw;
    cir.throw

    // if (b == 0)
    //   throw "Division by zero condition!";

    // Type info for char const*
    cir.global "private" constant external @_ZTIPKc : !cir.ptr<!u8i>
    cir.if %cond {
      %exception_addr = cir.alloc_exception 8 -> !cir.ptr<!void>
      ...
      // Store string addr for "Division by zero condition!"
      cir.store %string_addr, %exception_addr : !cir.ptr<!s8i>,
        !cir.ptr<!cir.ptr<!s8i>>
      cir.throw %exception_addr : !cir.ptr<!cir.ptr<!u8i>>,
        @_ZTIPKc
    ```
  }];

  let arguments = (ins
    Optional<CIR_PointerType>:$exception_ptr,
    OptionalAttr<FlatSymbolRefAttr>:$type_info,
    OptionalAttr<FlatSymbolRefAttr>:$dtor
  );

  let assemblyFormat = [{
    ($exception_ptr^ `:` type($exception_ptr))?
    (`,` $type_info^)?
    (`,` $dtor^)?
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool rethrows() { return getNumOperands() == 0; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AllocExceptionOp
//===----------------------------------------------------------------------===//

def CIR_AllocExceptionOp : CIR_Op<"alloc.exception"> {
  let summary = "Allocates an exception according to Itanium ABI";
  let description = [{
    Implements a slightly higher level __cxa_allocate_exception:

    `void *__cxa_allocate_exception(size_t thrown_size);`

    If the operation fails, the program terminates rather than throw.

    Example:

    ```mlir
    // if (b == 0) {
    //   ...
    //   throw "...";
    cir.if %10 {
        %11 = cir.alloc_exception 8 -> !cir.ptr<!void>
        ... // store exception content into %11
        cir.throw %11 : !cir.ptr<!cir.ptr<!u8i>>, ...
    ```
  }];

  let arguments = (ins I64Attr:$size);
  let results = (outs Res<CIR_PointerType, "", [MemAlloc<DefaultResource>]>:$addr);

  let assemblyFormat = [{
    $size `->` qualified(type($addr)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

def CIR_TryOp : CIR_Op<"try",[
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments
]> {
  let summary = "C++ try block";
  let description = [{
    Holds the lexical scope of `try {}`. Note that resources used on catch
    clauses are usually allocated in the same parent as `cir.try`.

    `synthetic`: use `cir.try` to represent try/catches not originally
    present in the source code. For example, a synthetic `cir.try` region
    is created around the constructor call when `operator new` is used
    so that the memory allocated will be freed if the constructor throws
    an exception.

    `cleanup`: indicates that there are cleanups that must be performed
    when exiting the try region via exception, even if the exception is not
    caught.

    Example:

    ```mlir
    cir.try {
      cir.call exception @function() : () -> ()
      cir.yield
    } catch [type #cir.global_view<@_ZTIPf> : !cir.ptr<!u8i>] {
      ...
      cir.yield
    } unwind {
      cir.resume
    }
    ```
  }];

  let arguments = (ins
    UnitAttr:$synthetic,
    UnitAttr:$cleanup,
    DefaultValuedAttr<CIR_TryHandlerArrayAttr, "{}">:$handler_types
  );

  let regions = (region
    AnyRegion:$try_region,
    VariadicRegion<AnyRegion>:$handler_regions
  );

  let assemblyFormat = [{
    (`synthetic` $synthetic^)?
    (`cleanup` $cleanup^)?
    $try_region
    custom<TryHandlerRegions>($handler_regions, $handler_types)
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins
      "llvm::function_ref<void(mlir::OpBuilder &, "
        "mlir::Location)>":$tryBuilder,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
        "mlir::OperationState &)>":$handlersBuilder),
    [{
      assert(tryBuilder && "expected builder callback for 'cir.try' body");
      assert(handlersBuilder
        && "expected builder callback for 'handlers' body");

      OpBuilder::InsertionGuard guard($_builder);

      // Try body region
      mlir::Region *tryBodyRegion = $_state.addRegion();

      // Create try body region and set insertion point
      $_builder.createBlock(tryBodyRegion);
      tryBuilder($_builder, $_state.location);
      handlersBuilder($_builder, $_state.location, $_state);
    }]>
  ];

  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// CatchParamOp
//===----------------------------------------------------------------------===//

def CIR_CatchParamOp : CIR_Op<"catch_param", [HasParent<"cir::TryOp">]> {
  let summary = "Represents the catch clause formal parameter";
  let description = [{
    The `cir.catch_param` is used to retrieves the exception object inside
    the handler regions of `cir.try`.

    This operation is used only before the CFG flatterning pass.

    Example:

    ```mlir
    %exception = cir.catch_param : !cir.ptr<!void>
    ```
  }];

  let results = (outs Optional<CIR_AnyType>:$param);
  let assemblyFormat = [{
    (`:` qualified(type($param))^)?
    attr-dict
  }];

  let hasLLVMLowering = false;
}

//===----------------------------------------------------------------------===//
// Exception related: EhInflightOp
//===----------------------------------------------------------------------===//

def CIR_EhInflightOp : CIR_Op<"eh.inflight_exception"> {
  let summary = "Materialize the catch clause formal parameter";
  let description = [{
    `cir.eh.inflight_exception` returns two values:
      - `exception_ptr`: The exception pointer for the inflight exception
      - `type_id`: the type info index for the exception type
    This operation is expected to be the first operation in the unwind
    destination basic blocks of a `cir.try_call` operation.

    The `cleanup` attribute indicates that clean up code must be run before the
    values produced by this operation are used to dispatch the exception. This
    cleanup code must be executed even if the exception is not caught.
    This helps CIR to pass down more accurate information for LLVM lowering
    to landingpads.

    Example:

    ```mlir
    %exception_ptr, %type_id = cir.eh.inflight_exception
    %exception_ptr, %type_id = cir.eh.inflight_exception [@_ZTIi, @_ZTIPKc]
    %exception_ptr, %type_id = cir.eh.inflight_exception cleanup
    ``
  }];

  let arguments = (ins UnitAttr:$cleanup,
                       OptionalAttr<FlatSymbolRefArrayAttr>:$catch_type_list);
  let results = (outs CIR_VoidPtrType:$exception_ptr, CIR_UInt32:$type_id);
  let assemblyFormat = [{
    (`cleanup` $cleanup^)?
    ($catch_type_list^)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Exception related: EhTypeIdOp
//===----------------------------------------------------------------------===//

def CIR_EhTypeIdOp : CIR_Op<"eh.typeid",
  [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Compute exception type id from its global type symbol";
  let description = [{
    Returns the exception type id for a given global symbol representing
    a type.

    Example:
    ```mlir
    %type_id = cir.eh.typeid @_ZTIi
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$type_sym);
  let results = (outs CIR_UInt32:$type_id);
  let assemblyFormat = [{
    $type_sym attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

def CIR_AtomicFetchKind : CIR_I32EnumAttr<
  "AtomicFetchKind", "Binary opcode for atomic fetch-and-update operations", [
    I32EnumAttrCase<"Add", 0, "add">,
    I32EnumAttrCase<"Sub", 1, "sub">,
    I32EnumAttrCase<"And", 2, "and">,
    I32EnumAttrCase<"Xor", 3, "xor">,
    I32EnumAttrCase<"Or", 4, "or">,
    I32EnumAttrCase<"Nand", 5, "nand">,
    I32EnumAttrCase<"Max", 6, "max">,
    I32EnumAttrCase<"Min", 7, "min">
]>;

def CIR_AtomicFetchOp : CIR_Op<"atomic.fetch", [
  AllTypesMatch<["result", "val"]>,
  TypesMatchWith<"type of 'val' must match the pointee type of 'ptr'",
    "ptr", "val", "mlir::cast<cir::PointerType>($_self).getPointee()">
]> {
  let summary = "Atomic fetch-and-update operation";
  let description = [{
    C/C++ atomic fetch-and-update operation. This operation implements the C/C++
    builtin functions `__atomic_<binop>_fetch`, `__atomic_fetch_<binop>`, and
    `__c11_atomic_fetch_<binop>`, where `<binop>` is one of the following binary
    opcodes: `add`, `sub`, `and`, `xor`, `or`, `nand`, `max`, and `min`.

    This operation takes 2 arguments: a pointer `ptr` and a value `val`. The
    type of `val` must match the pointee type of `ptr`. If the binary operation
    is `add`, `sub`, `max`, or `min`, the type of `val` may either be an integer
    type or a floating-point type. Otherwise, `val` must be an integer.

    This operation atomically loads the value from `ptr`, performs the binary
    operation as indicated by `binop` on the loaded value and `val`, and stores
    the result back to `ptr`. If the `fetch_first` flag is present, the result
    of this operation is the old value loaded from `ptr` before the binary
    operation. Otherwise, the result of this operation is the result of the
    binary operation.

    Example:
    %res = cir.atomic.fetch add seq_cst %ptr, %val
        : (!cir.ptr<!s32i>, !s32i) -> !s32i
  }];
  let results = (outs CIR_AnyIntOrFloatType:$result);
  let arguments = (ins
    Arg<CIR_PtrToIntOrFloatType, "", [MemRead, MemWrite]>:$ptr,
    CIR_AnyIntOrFloatType:$val,
    CIR_AtomicFetchKind:$binop,
    Arg<CIR_MemOrder, "memory order">:$mem_order,
    UnitAttr:$is_volatile,
    UnitAttr:$fetch_first
  );

  let assemblyFormat = [{
    $binop $mem_order
    (`fetch_first` $fetch_first^)?
    $ptr `,` $val
    (`volatile` $is_volatile^)?
    `:` `(` qualified(type($ptr)) `,` qualified(type($val)) `)`
    `->` type($result) attr-dict
  }];

  let hasVerifier = 1;

  let extraLLVMLoweringPatternDecl = [{
    mlir::Value buildPostOp(cir::AtomicFetchOp op, OpAdaptor adaptor,
                            mlir::ConversionPatternRewriter &rewriter,
                            mlir::Value rmwVal, bool isInt) const;

    mlir::Value buildMinMaxPostOp(cir::AtomicFetchOp op, OpAdaptor adaptor,
                                  mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Value rmwVal, bool isInt,
                                  bool isSigned) const;
  }];
}

def CIR_AtomicXchgOp : CIR_Op<"atomic.xchg", [
  AllTypesMatch<["result", "val"]>,
  TypesMatchWith<"type of 'val' must match the pointee type of 'ptr'",
    "ptr", "val", "mlir::cast<cir::PointerType>($_self).getPointee()">
]> {
  let summary = "Atomic exchange";
  let description = [{
    C/C++ atomic exchange operation. This operation implements the C/C++
    builtin function `__atomic_exchange`, `__atomic_exchange_n`, and
    `__c11_atomic_exchange`.

    This operation takes two arguments: a pointer `ptr` and a value `val`. The
    operation atomically replaces the value of the object pointed-to by `ptr`
    with `val`, and returns the original value of the object.

    Example:

    ```mlir
    %res = cir.atomic.xchg seq_cst %ptr, %val : !cir.ptr<!u64i> -> !u64i
    ```
  }];

  let results = (outs CIR_AnyType:$result);
  let arguments = (ins
    Arg<CIR_PointerType, "", [MemRead, MemWrite]>:$ptr,
    CIR_AnyType:$val,
    Arg<CIR_MemOrder, "memory order">:$mem_order,
    UnitAttr:$is_volatile
  );

  let assemblyFormat = [{
    $mem_order (`volatile` $is_volatile^)?
    $ptr `,` $val
    `:` functional-type(operands, results) attr-dict
  }];
}

def CIR_AtomicCmpXchgOp : CIR_Op<"atomic.cmpxchg", [
  AllTypesMatch<["old", "expected", "desired"]>,
  TypesMatchWith<"type of 'expected' must match the pointee type of 'ptr'",
    "ptr", "expected", "mlir::cast<cir::PointerType>($_self).getPointee()">,
  TypesMatchWith<"type of 'desired' must match the pointee type of 'ptr'",
    "ptr", "desired", "mlir::cast<cir::PointerType>($_self).getPointee()">
]> {
  let summary = "Atomic compare and exchange";
  let description = [{
    C/C++ atomic compare and exchange operation. Implements builtins like
    `__atomic_compare_exchange_n` and `__atomic_compare_exchange`.

    This operation takes three arguments: a pointer `ptr` and two values
    `expected` and `desired`. This operation compares the value of the object
    pointed-to by `ptr` with `expected`, and if they are equal, it sets the
    value of the object to `desired`.

    The `succ_order` attribute gives the memory order of this atomic operation
    when the exchange takes place. The `fail_order` attribute gives the memory
    order of this atomic operation when the exchange does not take place.

    The `weak` attribute is a boolean flag that indicates whether this is a
    "weak" compare-and-exchange operation. A weak compare-and-exchange operation
    allows "spurious failures", meaning that be treated as if the comparison
    failed and not exchange values even if `*ptr` and `expected` indeed compare
    equal.
    
    The type of `expected` and `desired` must be the same. The pointee type of
    `ptr` must be the same as the type of `expected` and `desired`.

    This operation has two results. The first result `old` gives the old value
    of the object pointed-to by `ptr`, regardless of whether the exchange
    actually took place. The second result `success` is a boolean flag
    indicating whether the exchange actually took place.

    Example:

    ```mlir
    %old, %success = cir.atomic.cmpxchg weak success(seq_cst) failure(acquire)
        %ptr, %expected, %desired
        : (!cir.ptr<!u64i>, !u64i, !u64i) -> (!u64i, !cir.bool)
    ```
  }];
  let results = (outs CIR_AnyType:$old, CIR_BoolType:$success);
  let arguments = (ins Arg<CIR_PointerType, "", [MemRead, MemWrite]>:$ptr,
                       CIR_AnyType:$expected,
                       CIR_AnyType:$desired,
                       Arg<CIR_MemOrder, "success memory order">:$succ_order,
                       Arg<CIR_MemOrder, "failure memory order">:$fail_order,
                       OptionalAttr<I64Attr>:$alignment,
                       UnitAttr:$weak,
                       UnitAttr:$is_volatile);

  let assemblyFormat = [{
    (`weak` $weak^)?
    `success` `(` $succ_order `)` `failure` `(` $fail_order `)`
    $ptr `,` $expected `,` $desired
    (`align` `(` $alignment^ `)`)?
    (`volatile` $is_volatile^)?
    `:` functional-type(operands, results) attr-dict
  }];
}

def CIR_AtomicTestAndSetOp : CIR_Op<"atomic.test_and_set"> {
  let summary = "Atomic test and set";
  let description = [{
    C/C++ atomic test and set operation. Implements the builtin function
    `__atomic_test_and_set`.

    The operation takes as its only operand a pointer to an 8-bit signed
    integer. The operation atomically set the integer to an implementation-
    defined non-zero "set" value. The result of the operation is a boolean value
    indicating whether the previous value of the integer was the "set" value.

    Example:
    ```mlir
      %res = cir.atomic.test_and_set seq_cst %ptr : !cir.ptr<!s8i> -> !cir.bool
    ```
  }];

  let arguments = (ins
    Arg<CIR_PtrToType<CIR_SInt8>, "", [MemRead, MemWrite]>:$ptr,
    Arg<CIR_MemOrder, "memory order">:$mem_order,
    OptionalAttr<I64Attr>:$alignment,
    UnitAttr:$is_volatile
  );

  let results = (outs CIR_BoolType:$result);

  let assemblyFormat = [{
    $mem_order $ptr
    (`volatile` $is_volatile^)?
    `:` qualified(type($ptr)) `->` qualified(type($result)) attr-dict
  }];
}

def CIR_AtomicClearOp : CIR_Op<"atomic.clear"> {
  let summary = "Atomic clear";
  let description = [{
    C/C++ atomic clear operation. Implements the builtin function
    `__atomic_clear`.

    The operation takes as its only operand a pointer to an 8-bit signed
    integer. The operation atomically sets the integer to zero.

    Example:
    ```mlir
      cir.atomic.clear seq_cst %ptr : !cir.ptr<!s8i>
    ```
  }];

  let arguments = (ins
    Arg<CIR_PtrToType<CIR_SInt8>, "", [MemRead, MemWrite]>:$ptr,
    Arg<CIR_MemOrder, "memory order">:$mem_order,
    OptionalAttr<I64Attr>:$alignment,
    UnitAttr:$is_volatile
  );

  let assemblyFormat = [{
    $mem_order $ptr
    (`volatile` $is_volatile^)?
    `:` qualified(type($ptr)) attr-dict
  }];
}

def CIR_AtomicFenceOp : CIR_Op<"atomic.fence"> {
  let summary = "Atomic thread fence";
  let description = [{
    C/C++ Atomic thread fence synchronization primitive. Implements the builtin
    `__atomic_thread_fence` which enforces memory ordering constraints across
    threads within the specified synchronization scope.

    This handles all variations including:
      - `__atomic_thread_fence`
      - `__atomic_signal_fence`
      - `__c11_atomic_thread_fence`
      - `__c11_atomic_signal_fence`

    Example:
    ```mlir
      cir.atomic.fence syncscope(system) seq_cst
      cir.atomic.fence syncscope(single_thread) seq_cst
    ```
  }];

  let arguments = (ins
    Arg<CIR_MemOrder, "memory order">:$ordering,
    OptionalAttr<CIR_SyncScopeKind>:$syncscope
  );

  let assemblyFormat = [{
    (`syncscope` `(` $syncscope^ `)`)? $ordering attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BlockAddressOp
//===----------------------------------------------------------------------===//

def CIR_BlockAddressOp : CIR_Op<"block_address", [Pure]> {
  let summary = "Get the address of a cir.label within a function";
  let description = [{
    The `cir.blockaddress` operation takes a function name and a label and
    produces a pointer value that represents the address of that cir.label
    within the specified function.

    This operation models GCC's "labels as values" extension (`&&label`), which
    allows taking the address of a local label and using it as a computed
    jump target (e.g., with `goto *addr;`).

    Example:
    ```mlir
    %1 = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init]
                                                          {alignment = 8 : i64}
    %addr = cir.block_address <@c, "label1"> : !cir.ptr<!cir.void>
    cir.store align(8) %addr, %1 : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
    cir.br ^bb1
   ^bb1:
    cir.label "label"
    ```
  }];

  let arguments = (ins CIR_BlockAddrInfoAttr:$block_addr_info);
  let results = (outs CIR_VoidPtrType:$addr);
  let assemblyFormat = [{
    $block_addr_info `:` qualified(type($addr)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CpuIdOp
//===----------------------------------------------------------------------===//

def CIR_CpuIdOp : CIR_Op<"cpuid"> {
  let summary = "Get information about the CPU";
  let description = [{
    The `cir.cpuid` operation retrieves different types of CPU information and
    stores it in an array of 4 integers.

    This operation takes 3 arguments: `cpu_info`, a pointer to an array of 4
    integers; `function_id`, an integer determining what type of information to
    be retrieved (for instance, basic information, processor information and
    features, or cache/TLB information); and `sub_function_id`, an integer that
    adds more detail about what information is requested.

    As a result, the array of 4 integers is filled with the requested
    information.

    Example:

    ```mlir
    cir.cpuid %cpui_info, %function_id, %sub_function_id : (!cir.ptr<!s32i>,
        !s32i, !s32i)
    ```
  }];

  let arguments = (ins
    Arg<CIR_PtrToType<CIR_SInt32>, "array address", [MemWrite]>:$cpu_info,
    CIR_SInt32:$function_id,
    CIR_SInt32:$sub_function_id
  );

  let assemblyFormat = [{
    $cpu_info`,` $function_id`,` $sub_function_id `:`
        qualified(type($cpu_info)) `,`
        type($function_id) `,`
        type($sub_function_id) attr-dict
    }];
}

#endif // CLANG_CIR_DIALECT_IR_CIROPS_TD
