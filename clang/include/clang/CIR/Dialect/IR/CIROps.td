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

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIROPS
#define LLVM_CLANG_CIR_DIALECT_IR_CIROPS

include "clang/CIR/Dialect/IR/CIRDialect.td"
include "clang/CIR/Dialect/IR/CIRTypes.td"
include "clang/CIR/Dialect/IR/CIRAttrs.td"

include "clang/CIR/Interfaces/ASTAttrInterfaces.td"
include "clang/CIR/Interfaces/CIROpInterfaces.td"
include "clang/CIR/Interfaces/CIRLoopOpInterface.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// CIR Ops
//===----------------------------------------------------------------------===//

class CIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CIR Op Traits
//===----------------------------------------------------------------------===//

def SameFirstOperandAndResultType :
  NativeOpTrait<"SameFirstOperandAndResultType">;
def SameSecondOperandAndResultType :
  NativeOpTrait<"SameSecondOperandAndResultType">;
def SameFirstSecondOperandAndResultType :
  NativeOpTrait<"SameFirstSecondOperandAndResultType">;

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

// The enumaration value isn't in sync with clang.
def CK_IntegralToBoolean : I32EnumAttrCase<"int_to_bool", 1>;
def CK_ArrayToPointerDecay : I32EnumAttrCase<"array_to_ptrdecay", 2>;
def CK_IntegralCast : I32EnumAttrCase<"integral", 3>;
def CK_BitCast : I32EnumAttrCase<"bitcast", 4>;
def CK_FloatingCast : I32EnumAttrCase<"floating", 5>;
def CK_PtrToBoolean : I32EnumAttrCase<"ptr_to_bool", 6>;
def CK_FloatToIntegral : I32EnumAttrCase<"float_to_int", 7>;
def CK_IntegralToPointer : I32EnumAttrCase<"int_to_ptr", 8>;
def CK_PointerToIntegral : I32EnumAttrCase<"ptr_to_int", 9>;
def CK_FloatToBoolean : I32EnumAttrCase<"float_to_bool", 10>;
def CK_BooleanToIntegral : I32EnumAttrCase<"bool_to_int", 11>;
def CK_IntegralToFloat : I32EnumAttrCase<"int_to_float", 12>;
def CK_BooleanToFloat : I32EnumAttrCase<"bool_to_float", 13>;

def CastKind : I32EnumAttr<
    "CastKind",
    "cast kind",
    [CK_IntegralToBoolean, CK_ArrayToPointerDecay, CK_IntegralCast,
     CK_BitCast, CK_FloatingCast, CK_PtrToBoolean, CK_FloatToIntegral,
     CK_IntegralToPointer, CK_PointerToIntegral, CK_FloatToBoolean,
     CK_BooleanToIntegral, CK_IntegralToFloat, CK_BooleanToFloat]> {
  let cppNamespace = "::mlir::cir";
}

def CastOp : CIR_Op<"cast", [Pure]> {
  // FIXME: not all conversions are free of side effects.
  let summary = "Conversion between values of different types";
  let description = [{
    Apply C/C++ usual conversions rules between values. Currently supported kinds:

    - `array_to_ptrdecay`
    - `bitcast`
    - `integral`
    - `int_to_bool`
    - `int_to_float`
    - `floating`
    - `float_to_int`
    - `float_to_bool`
    - `ptr_to_int`
    - `ptr_to_bool`
    - `bool_to_int`
    - `bool_to_float`

    This is effectively a subset of the rules from
    `llvm-project/clang/include/clang/AST/OperationKinds.def`; but note that some
    of the conversions aren't implemented in terms of `cir.cast`, `lvalue-to-rvalue`
    for instance is modeled as a regular `cir.load`.

    ```mlir
    %4 = cir.cast (int_to_bool, %3 : i32), !cir.bool
    ...
    %x = cir.cast(array_to_ptrdecay, %0 : !cir.ptr<!cir.array<i32 x 10>>), !cir.ptr<i32>
    ```
  }];

  let arguments = (ins CastKind:$kind, CIR_AnyType:$src);
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
// ObjSizeOp
//===----------------------------------------------------------------------===//

def SizeInfoTypeMin : I32EnumAttrCase<"min", 0>;
def SizeInfoTypeMax : I32EnumAttrCase<"max", 1>;

def SizeInfoType : I32EnumAttr<
    "SizeInfoType",
    "size info type",
    [SizeInfoTypeMin, SizeInfoTypeMax]> {
  let cppNamespace = "::mlir::cir";
}

def ObjSizeOp : CIR_Op<"objsize", [Pure]> {
  let summary = "Conversion between values of different types";
  let description = [{
  }];

  let arguments = (ins CIR_PointerType:$ptr, SizeInfoType:$kind,
                   UnitAttr:$dynamic);
  let results = (outs CIR_IntType:$result);

  let assemblyFormat = [{
    `(`
        $ptr `:` type($ptr) `,`
        $kind
        (`,` `dynamic` $dynamic^)?
    `)`
    `->` type($result) attr-dict
  }];

  // Nothing to verify that isn't already covered by constraints.
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// PtrDiffOp
//===----------------------------------------------------------------------===//

def PtrDiffOp : CIR_Op<"ptr_diff", [Pure, SameTypeOperands]> {

  let summary = "Pointer subtraction arithmetic";
  let description = [{
    `cir.ptr_diff` performs a subtraction between two pointer types with the
    same element type and produces a `mlir::cir::IntType` result.

    Note that the result considers the pointer size according to the ABI for
    the pointee sizes, e.g. the subtraction between two `!cir.ptr<!u64i>` might
    yield 1, meaning 8 bytes, whereas for `void` or function type pointees,
    yielding 8 means 8 bytes.

    ```mlir
    %7 = "cir.ptr_diff"(%0, %1) : !cir.ptr<!u64i> -> !u64i
    ```
  }];

  let results = (outs CIR_IntType:$result);
  let arguments = (ins CIR_PointerType:$lhs, CIR_PointerType:$rhs);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs  `)` `:` qualified(type($lhs)) `->` qualified(type($result)) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// PtrStrideOp
//===----------------------------------------------------------------------===//

def PtrStrideOp : CIR_Op<"ptr_stride",
                         [Pure, SameFirstOperandAndResultType]> {
  let summary = "Pointer access with stride";
  let description = [{
    Given a base pointer as first operand, provides a new pointer after applying
    a stride (second operand).

    ```mlir
    %3 = cir.const(0 : i32) : i32
    %4 = cir.ptr_stride(%2 : !cir.ptr<i32>, %3 : i32), !cir.ptr<i32>
    ```
  }];

  let arguments = (ins CIR_PointerType:$base, CIR_IntType:$stride);
  let results = (outs CIR_PointerType:$result);

  let assemblyFormat = [{
    `(` $base `:` qualified(type($base)) `,` $stride `:` qualified(type($stride)) `)`
    `,` qualified(type($result)) attr-dict
  }];

  let extraClassDeclaration = [{
    // Get type pointed by the base pointer.
    mlir::Type getElementTy() {
      return getBase().getType().cast<mlir::cir::PointerType>().getPointee();
    }
  }];

  // SameFirstOperandAndResultType already checks all we need.
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : CIR_Op<"const",
    [ConstantLike, Pure]> {
  // FIXME: Use SameOperandsAndResultType or similar and prevent eye bleeding
  // type repetition in the assembly form.

  let summary = "Defines a CIR constant";
  let description = [{
    The `cir.const` operation turns a literal into an SSA value. The data is
    attached to the operation as an attribute.

    ```mlir
      %0 = cir.const(42 : i32) : i32
      %1 = cir.const(4.2 : f32) : f32
      %2 = cir.const(nullptr : !cir.ptr<i32>) : !cir.ptr<i32>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins TypedAttrInterface:$value);

  // The constant operation returns a single value of CIR_AnyType.
  let results = (outs CIR_AnyType:$res);

  let assemblyFormat = [{
    `(` custom<ConstantValue>($value) `)` attr-dict `:` type($res)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
     bool isNullPtr() {
      if (const auto ptrAttr = getValue().dyn_cast<mlir::cir::ConstPtrAttr>())
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
                 "$_self.cast<PointerType>().getPointee()">]> {
  let summary = "Defines a scope-local variable";
  let description = [{
    The `cir.alloca` operation defines a scope-local variable.

    The presence `init` attribute indicates that the local variable represented
    by this alloca was originally initialized in C/C++ source code. In such
    cases, the first use contains the initialization (a cir.store, a cir.call
    to a ctor, etc).

    The `dynAllocSize` specifies the size to dynamically allocate on the stack
    and ignores the allocation size based on the original type. This is useful
    when handling VLAs and is omitted when declaring regular local variables.

    The result type is a pointer to the input's type.

    Example:

    ```mlir
    // int count = 3;
    %0 = cir.alloca i32, !cir.ptr<i32>, ["count", init] {alignment = 4 : i64}

    // int *ptr;
    %1 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["ptr"] {alignment = 8 : i64}
    ...
    ```
  }];

  let arguments = (ins
    Optional<CIR_IntType>:$dynAllocSize,
    TypeAttr:$allocaType,
    StrAttr:$name,
    UnitAttr:$init,
    ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment,
    OptionalAttr<ASTVarDeclInterface>:$ast
  );

  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$addr, "Type":$allocaType,
                   "StringRef":$name,
                   "IntegerAttr":$alignment)>,

    OpBuilder<(ins "Type":$addr,
                   "Type":$allocaType,
                   "StringRef":$name,
                   "IntegerAttr":$alignment,
                   "Value":$dynAllocSize),
    [{
      if (dynAllocSize)
        $_state.addOperands(dynAllocSize);
      build($_builder, $_state, addr, allocaType, name, alignment);
    }]>
  ];

  let extraClassDeclaration = [{
    // Whether the alloca input type is a pointer.
    bool isPointerType() { return getAllocaType().isa<::mlir::cir::PointerType>(); }

    bool isDynamic() { return (bool)getDynAllocSize(); }
  }];

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat = [{
    $allocaType `,` `cir.ptr` type($addr) `,`
    ($dynAllocSize^ `:` type($dynAllocSize) `,`)?
    `[` $name
       (`,` `init` $init^)?
    `]`
    (`ast` $ast^)? attr-dict
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : CIR_Op<"load", [
  TypesMatchWith<"type of 'result' matches pointee type of 'addr'",
                 "addr", "result",
                 "$_self.cast<PointerType>().getPointee()">]> {

  let summary = "Load value from memory adddress";
  let description = [{
    `cir.load` reads a value (lvalue to rvalue conversion) given an address
    backed up by a `cir.ptr` type. A unit attribute `deref` can be used to
    mark the resulting value as used by another operation to dereference
    a pointer. A unit attribute `volatile` can be used to indicate a volatile
    loading.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = cir.load %0 : !cir.ptr<i32>, i32

    // Load address from memory at address %0. %3 is used by at least one
    // operation that dereferences a pointer.
    %3 = cir.load deref %0 : cir.ptr <!cir.ptr<i32>>

    // Perform a volatile load from address in %0.
    %4 = cir.load volatile %0 : !cir.ptr<i32>, i32
    ```
  }];

  let arguments = (ins Arg<CIR_PointerType, "the address to load from",
                           [MemRead]>:$addr, UnitAttr:$isDeref,
                       UnitAttr:$is_volatile);
  let results = (outs CIR_AnyType:$result);

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat = [{
    (`deref` $isDeref^)? (`volatile` $is_volatile^)?
    $addr `:` `cir.ptr` type($addr) `,` type($result) attr-dict
  }];

  // FIXME: add verifier.
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : CIR_Op<"store", [
  TypesMatchWith<"type of 'value' matches pointee type of 'addr'",
                 "addr", "value",
                 "$_self.cast<PointerType>().getPointee()">]> {

  let summary = "Store value to memory address";
  let description = [{
    `cir.store` stores a value (first operand) to the memory address specified
    in the second operand. A unit attribute `volatile` can be used to indicate
    a volatile store.

    Example:

    ```mlir
    // Store a function argument to local storage, address in %0.
    cir.store %arg0, %0 : i32, !cir.ptr<i32>

    // Perform a volatile store into memory location at the address in %0.
    cir.store volatile %arg0, %0 : i32, !cir.ptr<i32>
    ```
  }];

  let arguments = (ins CIR_AnyType:$value,
                       Arg<CIR_PointerType, "the address to store the value",
                           [MemWrite]>:$addr,
                       UnitAttr:$is_volatile);

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat = [{
    (`volatile` $is_volatile^)?
    $value `,` $addr attr-dict `:` type($value) `,` `cir.ptr` type($addr)
  }];

  // FIXME: add verifier.
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : CIR_Op<"return", [ParentOneOf<["FuncOp", "ScopeOp", "IfOp",
                                              "SwitchOp", "DoWhileOp",
                                              "WhileOp", "ForOp"]>,
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
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
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
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "The if-then-else operation";
  let description = [{
    The `cir.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand is a `cir.bool`
    type.

    Examples:

    ```mlir
    cir.if %b  {
      ...
    } else {
      ...
    }

    cir.if %c  {
      ...
    }

    cir.if %c  {
      ...
      cir.br ^a
    ^a:
      cir.yield
    }
    ```

    `cir.if` defines no values and the 'else' can be omitted. `cir.yield` must
    explicitly terminate the region if it has more than one block.
  }];
  let arguments = (ins CIR_BoolType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>
  ];
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
    operation. First argument is a `cir.bool` condition to evaluate, followed
    by two regions to execute (true or false). This is different from `cir.if`
    since each region is one block sized and the `cir.yield` closing the block
    scope should have one argument.

    Example:

    ```mlir
    // x = cond ? a : b;

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
  let regions = (region SizedRegion<1>:$trueRegion,
                        SizedRegion<1>:$falseRegion);
  let results = (outs Optional<CIR_AnyType>:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$cond,
      "function_ref<void(OpBuilder &, Location)>":$trueBuilder,
      "function_ref<void(OpBuilder &, Location)>":$falseBuilder)
      >
  ];

  // All constraints already verified elsewhere.
  let hasVerifier = 0;

  let assemblyFormat = [{
    `(` $cond `,`
      `true` $trueRegion `,`
      `false` $falseRegion
    `)` `:` functional-type(operands, results) attr-dict
  }];
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

     - When in the `cond` region of a `cir.loop`, it continues the loop
       if true, or exits it if false.
     - When in the `ready` region of a `cir.await`, it branches to the `resume`
       region when true, and to the `suspend` region when false.

    Example:

    ```mlir
    cir.loop for(cond : {
      cir.condition(%arg0) // Branches to `step` region or exits.
    }, step : {
      [...]
    }) {
      [...]
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
    ParentOneOf<["IfOp", "ScopeOp", "SwitchOp", "WhileOp", "ForOp", "AwaitOp",
                 "TernaryOp", "GlobalOp", "DoWhileOp", "CatchOp", "TryOp",
                 "ArrayCtor", "ArrayDtor"]>]> {
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
    The `cir.break` operation is used to cease the control flow to the parent
    operation, exiting its region's control flow. It is only allowed if it is
    within a breakable operation (loops and `switch`).
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
    The `cir.continue` operation is used to continue execution to the next
    iteration of a loop. It is only allowed within `cir.loop` regions.
  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Resume
//===----------------------------------------------------------------------===//

def ResumeOp : CIR_Op<"resume", [ReturnLike, Terminator,
                                 ParentOneOf<["CatchOp"]>]> {
  let summary = "Resumes execution after not catching exceptions";
  let description = [{
    The `cir.resume` operation terminates a region on `cir.catch`, "resuming"
    or continuing the unwind process.

    Examples:
    ```mlir
    cir.catch ... {
      ...
      fallback { cir.resume };
    }
    ```
  }];

  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

def ScopeOp : CIR_Op<"scope", [
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope,
       NoRegionArguments]> {
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

  let builders = [
    // Scopes for yielding values.
    OpBuilder<(ins
              "function_ref<void(OpBuilder &, Type &, Location)>":$scopeBuilder)>,
    // Scopes without yielding values.
    OpBuilder<(ins "function_ref<void(OpBuilder &, Location)>":$scopeBuilder)>
  ];
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

def UnaryOpKind_Inc   : I32EnumAttrCase<"Inc",   1, "inc">;
def UnaryOpKind_Dec   : I32EnumAttrCase<"Dec",   2, "dec">;
def UnaryOpKind_Plus  : I32EnumAttrCase<"Plus",  3, "plus">;
def UnaryOpKind_Minus : I32EnumAttrCase<"Minus", 4, "minus">;
def UnaryOpKind_Not   : I32EnumAttrCase<"Not",   5, "not">;

def UnaryOpKind : I32EnumAttr<
    "UnaryOpKind",
    "unary operation kind",
    [UnaryOpKind_Inc,
     UnaryOpKind_Dec,
     UnaryOpKind_Plus,
     UnaryOpKind_Minus,
     UnaryOpKind_Not,
     ]> {
  let cppNamespace = "::mlir::cir";
}

// FIXME: Pure won't work when we add overloading.
def UnaryOp : CIR_Op<"unary", [Pure, SameOperandsAndResultType]> {
  let summary = "Unary operations";
  let description = [{
    `cir.unary` performs the unary operation according to
    the specified opcode kind: [inc, dec, plus, minus, not].

    It requires one input operand and has one result, both types
    should be the same.

    ```mlir
    %7 = cir.unary(inc, %1) : i32 -> i32
    %8 = cir.unary(dec, %2) : i32 -> i32
    ```
  }];

  let results = (outs CIR_AnyType:$result);
  let arguments = (ins Arg<UnaryOpKind, "unary op kind">:$kind, Arg<CIR_AnyType>:$input);

  let assemblyFormat = [{
      `(` $kind `,` $input `)` `:` type($input) `,` type($result) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BinOp
//===----------------------------------------------------------------------===//

// FIXME: represent Commutative, Idempotent traits for appropriate binops
def BinOpKind_Mul : I32EnumAttrCase<"Mul", 1, "mul">;
def BinOpKind_Div : I32EnumAttrCase<"Div", 2, "div">;
def BinOpKind_Rem : I32EnumAttrCase<"Rem", 3, "rem">;
def BinOpKind_Add : I32EnumAttrCase<"Add", 4, "add">;
def BinOpKind_Sub : I32EnumAttrCase<"Sub", 5, "sub">;
def BinOpKind_And : I32EnumAttrCase<"And", 8, "and">;
def BinOpKind_Xor : I32EnumAttrCase<"Xor", 9, "xor">;
def BinOpKind_Or  : I32EnumAttrCase<"Or", 10, "or">;

def BinOpKind : I32EnumAttr<
    "BinOpKind",
    "binary operation (arith and logic) kind",
    [BinOpKind_Mul, BinOpKind_Div, BinOpKind_Rem,
     BinOpKind_Add, BinOpKind_Sub,
     BinOpKind_And, BinOpKind_Xor,
     BinOpKind_Or]> {
  let cppNamespace = "::mlir::cir";
}

// FIXME: Pure won't work when we add overloading.
def BinOp : CIR_Op<"binop", [Pure,
  SameTypeOperands, SameOperandsAndResultType]> {

  let summary = "Binary operations (arith and logic)";
  let description = [{
    cir.binop performs the binary operation according to
    the specified opcode kind: [mul, div, rem, add, sub,
    and, xor, or].

    It requires two input operands and has one result, all types
    should be the same.

    ```mlir
    %7 = cir.binop(add, %1, %2) : !s32i
    %7 = cir.binop(mul, %1, %2) : !u8i
    ```
  }];

  // TODO: get more accurate than CIR_AnyType
  let results = (outs CIR_AnyType:$result);
  let arguments = (ins Arg<BinOpKind, "binop kind">:$kind,
                       CIR_AnyType:$lhs, CIR_AnyType:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

def ShiftOp : CIR_Op<"shift", [Pure]> {
  let summary = "Shift";
  let description = [{
    Shift `left` or `right`, according to the first operand. Second operand is
    the shift target and the third the amount.

    ```mlir
    %7 = cir.shift(left, %1 : !u64i, %4 : !s32i) -> !u64i
    ```
  }];

  let results = (outs CIR_IntType:$result);
  let arguments = (ins CIR_IntType:$value, CIR_IntType:$amount,
                       UnitAttr:$isShiftleft);

  let assemblyFormat = [{
    `(`
      (`left` $isShiftleft^) : (`right`)?
      `,` $value `:` type($value)
      `,` $amount `:` type($amount)
    `)` `->` type($result) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def CmpOpKind_LT : I32EnumAttrCase<"lt", 1>;
def CmpOpKind_LE : I32EnumAttrCase<"le", 2>;
def CmpOpKind_GT : I32EnumAttrCase<"gt", 3>;
def CmpOpKind_GE : I32EnumAttrCase<"ge", 4>;
def CmpOpKind_EQ : I32EnumAttrCase<"eq", 5>;
def CmpOpKind_NE : I32EnumAttrCase<"ne", 6>;

def CmpOpKind : I32EnumAttr<
    "CmpOpKind",
    "compare operation kind",
    [CmpOpKind_LT, CmpOpKind_LE, CmpOpKind_GT,
     CmpOpKind_GE, CmpOpKind_EQ, CmpOpKind_NE]> {
  let cppNamespace = "::mlir::cir";
}

// FIXME: Pure might not work when we add overloading.
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

  // TODO: get more accurate than CIR_AnyType
  let results = (outs CIR_AnyType:$result);
  let arguments = (ins Arg<CmpOpKind, "cmp kind">:$kind,
                       CIR_AnyType:$lhs, CIR_AnyType:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) `,` type($result) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// BitsOp
//===----------------------------------------------------------------------===//

class CIR_BitOp<string mnemonic, TypeConstraint inputTy>
    : CIR_Op<mnemonic, [Pure]> {
  let arguments = (ins inputTy:$input);
  let results = (outs SInt32:$result);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` `:` type($result) attr-dict
  }];
}

def BitClrsbOp : CIR_BitOp<"bit.clrsb", SIntOfWidths<[32, 64]>> {
  let summary = "Get the number of leading redundant sign bits in the input";
  let description = [{
    Compute the number of leading redundant sign bits in the input integer.

    The input integer must be a signed integer. The most significant bit of the
    input integer is the sign bit. The `cir.bit.clrsb` operation returns the
    number of redundant sign bits in the input, that is, the number of bits
    following the most significant bit that are identical to it.

    The bit width of the input integer must be either 32 or 64.

    Examples:

    ```mlir
    !s32i = !cir.int<s, 32>

    // %0 = 0xDEADBEEF, 0b1101_1110_1010_1101_1011_1110_1110_1111
    %0 = cir.const(#cir.int<3735928559> : !s32i) : !s32i
    // %1 will be 1 because there is 1 bit following the most significant bit
    // that is identical to it.
    %1 = cir.bit.clrsb(%0 : !s32i) : !s32i

    // %2 = 1, 0b0000_0000_0000_0000_0000_0000_0000_0001
    %2 = cir.const(#cir.int<1> : !s32i) : !s32i
    // %3 will be 30
    %3 = cir.bit.clrsb(%2 : !s32i) : !s32i
    ```
  }];
}

def BitClzOp : CIR_BitOp<"bit.clz", UIntOfWidths<[16, 32, 64]>> {
  let summary = "Get the number of leading 0-bits in the input";
  let description = [{
    Compute the number of leading 0-bits in the input.

    The input integer must be an unsigned integer. The `cir.bit.clz` operation
    returns the number of consecutive 0-bits at the most significant bit
    position in the input.

    This operation invokes undefined behavior if the input value is 0.

    Example:

    ```mlir
    !s32i = !cir.int<s, 32>
    !u32i = !cir.int<u, 32>

    // %0 = 0b0000_0000_0000_0000_0000_0000_0000_1000
    %0 = cir.const(#cir.int<8> : !u32i) : !u32i
    // %1 will be 28
    %1 = cir.bit.clz(%0 : !u32i) : !s32i
    ```
  }];
}

def BitCtzOp : CIR_BitOp<"bit.ctz", UIntOfWidths<[16, 32, 64]>> {
  let summary = "Get the number of trailing 0-bits in the input";
  let description = [{
    Compute the number of trailing 0-bits in the input.

    The input integer must be an unsigned integer. The `cir.bit.ctz` operation
    returns the number of consecutive 0-bits at the least significant bit
    position in the input.

    This operation invokes undefined behavior if the input value is 0.

    Example:

    ```mlir
    !s32i = !cir.int<s, 32>
    !u32i = !cir.int<u, 32>

    // %0 = 0b1000
    %0 = cir.const(#cir.int<8> : !u32i) : !u32i
    // %1 will be 3
    %1 = cir.bit.ctz(%0 : !u32i) : !s32i
    ```
  }];
}

def BitFfsOp : CIR_BitOp<"bit.ffs", SIntOfWidths<[32, 64]>> {
  let summary = "Get the position of the least significant 1-bit of input";
  let description = [{
    Compute the position of the least significant 1-bit of the input.

    The input integer must be a signed integer. The `cir.bit.ffs` operation
    returns one plus the index of the least significant 1-bit of the input
    signed integer. As a special case, if the input integer is 0, `cir.bit.ffs`
    returns 0.

    Example:

    ```mlir
    !s32i = !cir.int<s, 32>

    // %0 = 0x0010_1000
    %0 = cir.const(#cir.int<40> : !s32i) : !s32i
    // #1 will be 4 since the 4th least significant bit is 1.
    %1 = cir.bit.ffs(%0 : !s32i) : !s32i
    ```
  }];
}

def BitParityOp : CIR_BitOp<"bit.parity", UIntOfWidths<[32, 64]>> {
  let summary = "Get the parity of input";
  let description = [{
    Compute the parity of the input. The parity of an integer is the number of
    1-bits in it modulo 2.

    The input must be an unsigned integer.

    Example:

    ```mlir
    !s32i = !cir.int<s, 32>
    !u32i = !cir.int<u, 32>

    // %0 = 0x0110_1000
    %0 = cir.const(#cir.int<104> : !u32i) : !s32i
    // %1 will be 1 since there are 3 1-bits in %0
    %1 = cir.bit.parity(%0 : !u32i) : !s32i
    ```
  }];
}

def BitPopcountOp : CIR_BitOp<"bit.popcount", UIntOfWidths<[16, 32, 64]>> {
  let summary = "Get the number of 1-bits in input";
  let description = [{
    Compute the number of 1-bits in the input.

    The input must be an unsigned integer.

    Example:

    ```mlir
    !s32i = !cir.int<s, 32>
    !u32i = !cir.int<u, 32>

    // %0 = 0x0110_1000
    %0 = cir.const(#cir.int<104> : !u32i) : !s32i
    // %1 will be 3 since there are 3 1-bits in %0
    %1 = cir.bit.popcount(%0 : !u32i) : !s32i
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ByteswapOp
//===----------------------------------------------------------------------===//

def ByteswapOp : CIR_Op<"bswap", [Pure, SameOperandsAndResultType]> {
  let summary = "Reverse the bytes that constitute the operand integer";
  let description = [{
    The `cir.bswap` operation takes an integer as operand, and returns it with
    the order of bytes that constitute the operand reversed.

    The operand integer must be an unsigned integer. Its widths must be either
    16, 32, or 64.

    Example:

    ```mlir
    !u32i = !cir.int<u, 32>

    // %0 = 0x12345678
    %0 = cir.const(#cir.int<305419896> : !u32i) : !u32i

    // %1 should be 0x78563412
    %1 = cir.bswap(%0 : !u32i) : !u32i
    ```
  }];

  let results = (outs CIR_IntType:$result);
  let arguments = (ins UIntOfWidths<[16, 32, 64]>:$input);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CmpThreeWayOp
//===----------------------------------------------------------------------===//

def CmpThreeWayOp : CIR_Op<"cmp3way", [Pure, SameTypeOperands]> {
  let summary = "Compare two values with C++ three-way comparison semantics";
  let description = [{
    The `cir.cmp3way` operation models the `<=>` operator in C++20. It takes two
    operands with the same type and produces a result indicating the ordering
    between the two input operands.

    The result of the operation is a signed integer that indicates the ordering
    between the two input operands.

    There are two kinds of ordering: strong ordering and partial ordering.
    Comparing different types of values yields different kinds of orderings.
    The `info` parameter gives the ordering kind and other necessary information
    about the comparison.

    Example:

    ```mlir
    !s32i = !cir.int<s, 32>

    #cmp3way_strong = #cmp3way_info<strong, lt = -1, eq = 0, gt = 1>
    #cmp3way_partial = #cmp3way_info<strong, lt = -1, eq = 0, gt = 1, unordered = 2>

    %0 = cir.const(#cir.int<0> : !s32i) : !s32i
    %1 = cir.const(#cir.int<1> : !s32i) : !s32i
    %2 = cir.cmp3way(%0 : !s32i, %1, #cmp3way_strong) : !s8i

    %3 = cir.const(#cir.fp<0.0> : !cir.float) : !cir.float
    %4 = cir.const(#cir.fp<1.0> : !cir.float) : !cir.float
    %5 = cir.cmp3way(%3 : !cir.float, %4, #cmp3way_partial) : !s8i
    ```
  }];

  let results = (outs CIR_IntType:$result);
  let arguments = (ins CIR_AnyType:$lhs, CIR_AnyType:$rhs,
                       CmpThreeWayInfoAttr:$info);

  let assemblyFormat = [{
    `(` $lhs `:` type($lhs) `,` $rhs `,` qualified($info) `)`
    `:` type($result) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

def CaseOpKind_DT : I32EnumAttrCase<"Default", 1, "default">;
def CaseOpKind_EQ : I32EnumAttrCase<"Equal", 2, "equal">;
def CaseOpKind_AO : I32EnumAttrCase<"Anyof", 3, "anyof">;

def CaseOpKind : I32EnumAttr<
    "CaseOpKind",
    "case kind",
    [CaseOpKind_DT, CaseOpKind_EQ, CaseOpKind_AO]> {
  let cppNamespace = "::mlir::cir";
}

def CaseEltValueListAttr :
  TypedArrayAttrBase<AnyAttr, "cir.switch case value condition"> {
  let constBuilderCall = ?;
}

def CaseAttr : AttrDef<CIR_Dialect, "Case"> {
  // FIXME: value should probably be optional for more clear "default"
  // representation.
  let parameters = (ins "ArrayAttr":$value, "CaseOpKindAttr":$kind);
  let mnemonic = "case";
  let assemblyFormat = "`<` struct(params) `>`";
}

def CaseArrayAttr :
  TypedArrayAttrBase<CaseAttr, "cir.switch case array attribute"> {
  let constBuilderCall = ?;
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

    A variadic list of "case" attribute operands and regions track the possible
    control flow within `cir.switch`. A `case` must be in one of the following forms:
    - `equal, <constant>`: equality of the second case operand against the
    condition.
    - `anyof, [constant-list]`: equals to any of the values in a subsequent
    following list.
    - `default`: any other value.

    Each case region must be explicitly terminated.

    Examples:

    ```mlir
    cir.switch (%b : i32) [
      case (equal, 20) {
        ...
        cir.yield break
      },
      case (anyof, [1, 2, 3] : i32) {
        ...
        cir.return ...
      }
      case (default) {
        ...
        cir.yield fallthrough
      }
    ]
    ```
  }];

  let arguments = (ins CIR_IntType:$condition,
                       OptionalAttr<CaseArrayAttr>:$cases);

  let regions = (region VariadicRegion<AnyRegion>:$regions);

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$condition,
               "function_ref<void(OpBuilder &, Location, OperationState &)>":$switchBuilder)>
  ];

  let assemblyFormat = [{
    custom<SwitchOp>(
      $regions, $cases, $condition, type($condition)
    )
    attr-dict
  }];
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

    Example:

    ```mlir
      ...
        cir.br ^bb3
      ^bb3:
        cir.return
    ```
  }];

  let builders = [
    OpBuilder<(ins "Block *":$dest,
              CArg<"ValueRange", "{}">:$destOperands), [{
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
// BrCondOp
//===----------------------------------------------------------------------===//

def BrCondOp : CIR_Op<"brcond",
      [DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
       Pure, Terminator, SameVariadicOperandSize]> {
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
    OpBuilder<(ins "Value":$cond, "Block *":$destTrue, "Block *":$destFalse,
               CArg<"ValueRange", "{}">:$destOperandsTrue,
               CArg<"ValueRange", "{}">:$destOperandsFalse), [{
      $_state.addOperands(cond);
      $_state.addSuccessors(destTrue);
      $_state.addSuccessors(destFalse);
      $_state.addOperands(destOperandsTrue);
      $_state.addOperands(destOperandsFalse);
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
// While & DoWhileOp
//===----------------------------------------------------------------------===//

class WhileOpBase<string mnemonic> : CIR_Op<mnemonic, [
  LoopOpInterface,
  NoRegionArguments,
]> {
  defvar isWhile = !eq(mnemonic, "while");
  let summary = "C/C++ " # !if(isWhile, "while", "do-while") # " loop";
  let builders = [
    OpBuilder<(ins "function_ref<void(OpBuilder &, Location)>":$condBuilder,
                   "function_ref<void(OpBuilder &, Location)>":$bodyBuilder), [{
        OpBuilder::InsertionGuard guard($_builder);
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
    Region &getEntry() { return getBody(); }
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

def ForOp : CIR_Op<"for", [LoopOpInterface, NoRegionArguments]> {
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
    OpBuilder<(ins "function_ref<void(OpBuilder &, Location)>":$condBuilder,
                   "function_ref<void(OpBuilder &, Location)>":$bodyBuilder,
                   "function_ref<void(OpBuilder &, Location)>":$stepBuilder), [{
        OpBuilder::InsertionGuard guard($_builder);

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
    Region *maybeGetStep() { return &getStep(); }
    llvm::SmallVector<Region *> getRegionsInExecutionOrder() {
      return llvm::SmallVector<Region *, 3>{&getCond(), &getBody(), &getStep()};
    }
  }];
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

// Linkage types. This is currently a replay of llvm/IR/GlobalValue.h, this is
// currently handy as part of forwarding appropriate linkage types for LLVM
// lowering, specially useful for C++ support.

// Externally visible function
def Global_ExternalLinkage :
  I32EnumAttrCase<"ExternalLinkage", 0, "external">;
// Available for inspection, not emission.
def Global_AvailableExternallyLinkage :
  I32EnumAttrCase<"AvailableExternallyLinkage", 1, "available_externally">;
// Keep one copy of function when linking (inline)
def Global_LinkOnceAnyLinkage :
  I32EnumAttrCase<"LinkOnceAnyLinkage", 2, "linkonce">;
// Same, but only replaced by something equivalent.
def Global_LinkOnceODRLinkage :
  I32EnumAttrCase<"LinkOnceODRLinkage", 3, "linkonce_odr">;
// Keep one copy of named function when linking (weak)
def Global_WeakAnyLinkage :
  I32EnumAttrCase<"WeakAnyLinkage", 4, "weak">;
// Same, but only replaced by something equivalent.
def Global_WeakODRLinkage :
  I32EnumAttrCase<"WeakODRLinkage", 5, "weak_odr">;
// TODO: should we add something like appending linkage too?
// Special purpose, only applies to global arrays
// def Global_AppendingLinkage :
//   I32EnumAttrCase<"AppendingLinkage", 6, "appending">;
// Rename collisions when linking (static functions).
def Global_InternalLinkage :
  I32EnumAttrCase<"InternalLinkage", 7, "internal">;
// Like Internal, but omit from symbol table, prefix it with
// "cir_" to prevent clash with MLIR's symbol "private".
def Global_PrivateLinkage :
  I32EnumAttrCase<"PrivateLinkage", 8, "cir_private">;
// ExternalWeak linkage description.
def Global_ExternalWeakLinkage :
  I32EnumAttrCase<"ExternalWeakLinkage", 9, "extern_weak">;
// Tentative definitions.
def Global_CommonLinkage :
  I32EnumAttrCase<"CommonLinkage", 10, "common">;

/// An enumeration for the kinds of linkage for global values.
def GlobalLinkageKind : I32EnumAttr<
    "GlobalLinkageKind",
    "Linkage type/kind",
    [Global_ExternalLinkage, Global_AvailableExternallyLinkage,
     Global_LinkOnceAnyLinkage, Global_LinkOnceODRLinkage,
     Global_WeakAnyLinkage, Global_WeakODRLinkage,
     Global_InternalLinkage, Global_PrivateLinkage,
     Global_ExternalWeakLinkage, Global_CommonLinkage
     ]> {
  let cppNamespace = "::mlir::cir";
}

def SOB_Undefined : I32EnumAttrCase<"undefined", 1>;
def SOB_Defined : I32EnumAttrCase<"defined", 2>;
def SOB_Trapping : I32EnumAttrCase<"trapping", 3>;

def SignedOverflowBehaviorEnum : I32EnumAttr<
    "SignedOverflowBehavior",
    "the behavior for signed overflow",
    [SOB_Undefined, SOB_Defined, SOB_Trapping]> {
  let cppNamespace = "::mlir::cir::sob";
}


def GlobalOp : CIR_Op<"global", [Symbol, DeclareOpInterfaceMethods<RegionBranchOpInterface>, NoRegionArguments]> {
  let summary = "Declares or defines a global variable";
  let description = [{
    The `cir.global` operation declares or defines a named global variable.

    The backing memory for the variable is allocated statically and is
    described by the type of the variable.

    The operation is a declaration if no `inital_value` is
    specified, else it is a definition.

    The global variable can also be marked constant using the
    `constant` unit attribute. Writing to such constant global variables is
    undefined.

    The `linkage` tracks C/C++ linkage types, currently very similar to LLVM's.
    Symbol visibility in `sym_visibility` is defined in terms of MLIR's visibility
    and verified to be in accordance to `linkage`.

    Example:

    ```mlir
    // Public and constant variable with initial value.
    cir.global public constant @c : i32 = 4;
    ```
  }];

  // Note that both sym_name and sym_visibility are tied to Symbol trait.
  // TODO: sym_visibility can possibly be represented by implementing the
  // necessary Symbol's interface in terms of linkage instead.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       TypeAttr:$sym_type,
                       Arg<GlobalLinkageKind, "linkage type">:$linkage,
                       // Note this can also be a FlatSymbolRefAttr
                       OptionalAttr<AnyAttr>:$initial_value,
                       UnitAttr:$constant,
                       OptionalAttr<I64Attr>:$alignment,
                       OptionalAttr<ASTVarDeclInterface>:$ast,
                       OptionalAttr<StrAttr>:$section
                       );
  let regions = (region AnyRegion:$ctorRegion, AnyRegion:$dtorRegion);
  let assemblyFormat = [{
       ($sym_visibility^)?
       (`constant` $constant^)?
       $linkage
       $sym_name
       custom<GlobalOpTypeAndInitialValue>($sym_type, $initial_value, $ctorRegion, $dtorRegion)
       attr-dict
  }];

  let extraClassDeclaration = [{
    bool isDeclaration() {
      return !getInitialValue() && getCtorRegion().empty() && getDtorRegion().empty();
    }
    bool hasInitializer() { return !isDeclaration(); }
    bool hasAvailableExternallyLinkage() {
      return mlir::cir::isAvailableExternallyLinkage(getLinkage());
    }
    bool isDeclarationForLinker() {
      if (hasAvailableExternallyLinkage())
        return true;

      return isDeclaration();
    }

    /// Whether the definition of this global may be replaced at link time.
    bool isWeakForLinker() { return cir::isWeakForLinker(getLinkage()); }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      // MLIR's default visibility is public.
      "StringRef":$sym_name,
      "Type":$sym_type,
      CArg<"bool", "false">:$isConstant,
      // CIR defaults to external linkage.
      CArg<"cir::GlobalLinkageKind",
            "cir::GlobalLinkageKind::ExternalLinkage">:$linkage,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$ctorBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$dtorBuilder)>
  ];

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
     undefined. Resulting type must always be a `!cir.ptr<...>` type.

     Example:

     ```mlir
     %x = cir.get_global @foo : !cir.ptr<i32>
     ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs Res<CIR_PointerType, "", []>:$addr);

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat = "$name `:` `cir.ptr` type($addr) attr-dict";

  // `GetGlobalOp` is fully verified by its traits.
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// VTableAddrPointOp
//===----------------------------------------------------------------------===//

def VTableAddrPointOp : CIR_Op<"vtable.address_point",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get the vtable (global variable) address point";
  let description = [{
    The `vtable.address_point` operation retrieves the "effective" address
    (address point) of a C++ virtual table. An object internal `__vptr`
    gets initializated on top of the value returned by this operation.

    `vtable_index` provides the appropriate vtable within the vtable group
    (as specified by Itanium ABI), and `addr_point_index` the actual address
    point within that vtable.

    The return type is always a `!cir.ptr<!cir.ptr<() -> i32>>`.

    Example:
    ```mlir
    cir.global linkonce_odr @_ZTV1B = ...
    ...
    %3 = cir.vtable.address_point(@_ZTV1B, vtable_index = 0, address_point_index = 2) : cir.ptr <!cir.ptr<() -> i32>>
    ```
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$name,
                       Optional<CIR_AnyType>:$sym_addr,
                       I32Attr:$vtable_index,
                       I32Attr:$address_point_index);
  let results = (outs Res<CIR_PointerType, "", []>:$addr);

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat = [{
    `(`
      ($name^)?
      ($sym_addr^ `:` type($sym_addr))?
      `,`
        `vtable_index` `=` $vtable_index `,`
        `address_point_index` `=` $address_point_index
    `)`
    `:` `cir.ptr` type($addr) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SetBitfieldOp
//===----------------------------------------------------------------------===//

def SetBitfieldOp : CIR_Op<"set_bitfield"> {
  let summary = "Set a bitfield";
  let description = [{
    The `cir.set_bitfield` operation provides a store-like access to
    a bit field of a record.

    It expects an address of a storage where to store, a type of the storage,
    a value being stored, a name of a bit field, a pointer to the storage in the
    base record, a size of the storage, a size the bit field, an offset
    of the bit field and a sign. Returns a value being stored.

    A unit attribute `volatile` can be used to indicate a volatile load of the
    bitfield.

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
      s.d = 3;
    }
    ```

    ```mlir
    // 'd' is in the storage with the index 1
    !struct_type = !cir.struct<struct "S" {!cir.int<u, 32>, !cir.int<u, 32>, !cir.int<u, 16>} #cir.record.decl.ast>
    #bfi_d = #cir.bitfield_info<name = "d", storage_type = !u32i, size = 2, offset = 17, is_signed = true>

    %1 = cir.const(#cir.int<3> : !s32i) : !s32i
    %2 = cir.load %0 : cir.ptr <!cir.ptr<!struct_type>>, !cir.ptr<!struct_type>
    %3 = cir.get_member %2[1] {name = "d"} : !cir.ptr<!struct_type> -> !cir.ptr<!u32i>
    %4 = cir.set_bitfield(#bfi_d, %3 : !cir.ptr<!u32i>, %1 : !s32i) -> !s32i
    ```
   }];

  let arguments = (ins
    Arg<CIR_PointerType, "the address to store the value", [MemWrite]>:$addr,
    CIR_AnyType:$src,
    BitfieldInfoAttr:$bitfield_info,
    UnitAttr:$is_volatile
  );

  let results = (outs CIR_IntType:$result);

  let assemblyFormat = [{ `(`$bitfield_info`,` $addr`:`qualified(type($addr))`,`
    $src`:`type($src) `)`  attr-dict `->` type($result) }];

  let builders = [
    OpBuilder<(ins "Type":$type,
                   "Value":$addr,
                   "Type":$storage_type,
                   "Value":$src,
                   "StringRef":$name,
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
      build($_builder, $_state, type, addr, src, info, is_volatile);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// GetBitfieldOp
//===----------------------------------------------------------------------===//

def GetBitfieldOp : CIR_Op<"get_bitfield"> {
  let summary = "Get a bitfield";
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
    different storages. The `cir.get_bitfield` operation gets the value
    of the bitfield
    ```C++
    typedef struct {
      int a : 4;
      int b : 27;
      int c : 17;
      int d : 2;
      int e : 15;
    } S;

    int load_bitfield(S& s) {
      return s.d;
    }
    ```

    ```mlir
    // 'd' is in the storage with the index 1
    !struct_type = !cir.struct<struct "S" {!cir.int<u, 32>, !cir.int<u, 32>, !cir.int<u, 16>} #cir.record.decl.ast>
    #bfi_d = #cir.bitfield_info<name = "d", storage_type = !u32i, size = 2, offset = 17, is_signed = true>

    %2 = cir.load %0 : cir.ptr <!cir.ptr<!struct_type>>, !cir.ptr<!struct_type>
    %3 = cir.get_member %2[1] {name = "d"} : !cir.ptr<!struct_type> -> !cir.ptr<!u32i>
    %4 = cir.get_bitfield(#bfi_d, %3 : !cir.ptr<!u32i>) -> !s32i
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
    OpBuilder<(ins "Type":$type,
                   "Value":$addr,
                   "Type":$storage_type,
                   "StringRef":$name,
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
  let summary = "Get the address of a member of a struct";
  let description = [{
    The `cir.get_member` operation gets the address of a particular named
    member from the input record.

    It expects a pointer to the base record as well as the name of the member
    and its field index.

    Example:
    ```mlir
    // Suppose we have a struct with multiple members.
    !s32i = !cir.int<s, 32>
    !s8i = !cir.int<s, 32>
    !struct_ty = !cir.struct<"struct.Bar" {!s32i, !s8i}>

    // Get the address of the member at index 1.
    %1 = cir.get_member %0[1] {name = "i"} : (!cir.ptr<!struct_ty>) -> !cir.ptr<!s8i>
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
    OpBuilder<(ins "Type":$type,
                   "Value":$value,
                   "llvm::StringRef":$name,
                   "unsigned":$index),
    [{
      mlir::APInt fieldIdx(64, index);
      build($_builder, $_state, type, value, name, fieldIdx);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the index of the struct member being accessed.
    uint64_t getIndex() { return getIndexAttr().getZExtValue(); }

    /// Return the record type pointed by the base pointer.
    mlir::cir::PointerType getAddrTy() { return getAddr().getType(); }

    /// Return the result type.
    mlir::cir::PointerType getResultTy() {
      return getResult().getType().cast<mlir::cir::PointerType>();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetRuntimeMemberOp
//===----------------------------------------------------------------------===//

def GetRuntimeMemberOp : CIR_Op<"get_runtime_member"> {
  let summary = "Get the address of a member of a struct";
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
    (void)f->*member;  // cir.get_runtime_member
    ```

    This operation expects a pointer to the base record as well as the pointer
    to the target member.
  }];

  let arguments = (ins
    Arg<StructPtr, "address of the struct object", [MemRead]>:$addr,
    Arg<CIR_DataMemberType, "pointer to the target member">:$member);

  let results = (outs Res<CIR_PointerType, "">:$result);

  let assemblyFormat = [{
    $addr `[` $member `:` qualified(type($member)) `]` attr-dict
    `:` qualified(type($addr)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VecInsertOp
//===----------------------------------------------------------------------===//

def VecInsertOp : CIR_Op<"vec.insert", [Pure,
  TypesMatchWith<"argument type matches vector element type", "vec", "value",
                 "$_self.cast<VectorType>().getEltType()">,
  AllTypesMatch<["result", "vec"]>]> {

  let summary = "Insert one element into a vector object";
  let description = [{
    The `cir.vec.insert` operation replaces the element of the given vector at
    the given index with the given value.  The new vector with the inserted
    element is returned.
  }];

  let arguments = (ins CIR_VectorType:$vec, AnyType:$value, CIR_IntType:$index);
  let results = (outs CIR_VectorType:$result);

  let assemblyFormat = [{
    $value `,` $vec `[` $index `:` type($index) `]` attr-dict `:`
    qualified(type($vec))
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// VecExtractOp
//===----------------------------------------------------------------------===//

def VecExtractOp : CIR_Op<"vec.extract", [Pure,
  TypesMatchWith<"type of 'result' matches element type of 'vec'", "vec",
                 "result", "$_self.cast<VectorType>().getEltType()">]> {

  let summary = "Extract one element from a vector object";
  let description = [{
    The `cir.vec.extract` operation extracts the element at the given index
    from a vector object.
  }];

  let arguments = (ins CIR_VectorType:$vec, CIR_IntType:$index);
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    $vec `[` $index `:` type($index) `]` attr-dict `:` qualified(type($vec))
  }];

  let hasVerifier = 0;
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

  let arguments = (ins Variadic<CIR_AnyType>:$elements);
  let results = (outs CIR_VectorType:$result);

  let assemblyFormat = [{
    `(` ($elements^ `:` type($elements))? `)` `:` qualified(type($result))
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VecSplat
//===----------------------------------------------------------------------===//

// cir.vec.splat is a separate operation from cir.vec.create because more
// efficient LLVM IR can be generated for it, and because some optimization and
// analysis passes can benefit from knowing that all elements of the vector
// have the same value.

def VecSplatOp : CIR_Op<"vec.splat", [Pure,
  TypesMatchWith<"type of 'value' matches element type of 'result'", "result",
                 "value", "$_self.cast<VectorType>().getEltType()">]> {

  let summary = "Convert a scalar into a vector";
  let description = [{
    The `cir.vec.splat` operation creates a vector value from a scalar value.
    All elements of the vector have the same value, that of the given scalar.
  }];

  let arguments = (ins CIR_AnyType:$value);
  let results = (outs CIR_VectorType:$result);

  let assemblyFormat = [{
    $value `:` type($value) `,` qualified(type($result)) attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// VecCmp
//===----------------------------------------------------------------------===//

def VecCmpOp : CIR_Op<"vec.cmp", [Pure, SameTypeOperands]> {

  let summary = "Compare two vectors";
  let description = [{
    The `cir.vec.cmp` operation does an element-wise comparison of two vectors
    of the same type. The result is a vector of the same size as the operands
    whose element type is the signed integral type that is the same size as the
    element type of the operands.  The values in the result are 0 or -1.
  }];

  let arguments = (ins Arg<CmpOpKind, "cmp kind">:$kind, CIR_VectorType:$lhs,
                       CIR_VectorType:$rhs);
  let results = (outs CIR_VectorType:$result);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs `)` `:` qualified(type($lhs)) `,`
    qualified(type($result)) attr-dict
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// VecTernary
//===----------------------------------------------------------------------===//

def VecTernaryOp : CIR_Op<"vec.ternary",
                   [Pure, AllTypesMatch<["result", "vec1", "vec2"]>]> {
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
  let arguments = (ins IntegerVector:$cond, CIR_VectorType:$vec1,
		       CIR_VectorType:$vec2);
  let results = (outs CIR_VectorType:$result);
  let assemblyFormat = [{
    `(` $cond `,` $vec1 `,` $vec2 `)` `:` qualified(type($cond)) `,`
    qualified(type($vec1)) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VecShuffle
//===----------------------------------------------------------------------===//

// TODO: Create an interface that both VecShuffleOp and VecShuffleDynamicOp
// implement.  This could be useful for passes that don't care how the vector
// shuffle was specified.

def VecShuffleOp : CIR_Op<"vec.shuffle",
                   [Pure, AllTypesMatch<["vec1", "vec2"]>]> {
  let summary = "Combine two vectors using indices passed as constant integers";
  let description = [{
    The `cir.vec.shuffle` operation implements the documented form of Clang's
    __builtin_shufflevector, where the indices of the shuffled result are
    integer constants.

    The two input vectors, which must have the same type, are concatenated.
    Each of the integer constant arguments is interpreted as an index into that
    concatenated vector, with a value of -1 meaning that the result value
    doesn't matter. The result vector, which must have the same element type as
    the input vectors and the same number of elements as the list of integer
    constant indices, is constructed by taking the elements at the given
    indices from the concatenated vector.  The size of the result vector does
    not have to match the size of the individual input vectors or of the
    concatenated vector.
  }];
  let arguments = (ins CIR_VectorType:$vec1, CIR_VectorType:$vec2,
		       ArrayAttr:$indices);
  let results = (outs CIR_VectorType:$result);
  let assemblyFormat = [{
    `(` $vec1 `,` $vec2 `:` qualified(type($vec1)) `)` $indices `:`
     qualified(type($result)) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VecShuffleDynamic
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
  }];
  let arguments = (ins CIR_VectorType:$vec, IntegerVector:$indices);
  let results = (outs CIR_VectorType:$result);
  let assemblyFormat = [{
    $vec `:` qualified(type($vec)) `,` $indices `:` qualified(type($indices))
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BaseClassAddr
//===----------------------------------------------------------------------===//

def BaseClassAddrOp : CIR_Op<"base_class_addr"> {
  let summary = "Get the base class address for a class/struct";
  let description = [{
    The `cir.base_class_addr` operaration gets the address of a particular
    base class given a derived class pointer.

    Example:
    ```mlir
    TBD
    ```
  }];

  let arguments = (ins
    Arg<CIR_PointerType, "derived class pointer", [MemRead]>:$derived_addr);

  let results = (outs Res<CIR_PointerType, "">:$base_addr);

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat = [{
    `(`
      $derived_addr `:` `cir.ptr` type($derived_addr)
    `)` `->` `cir.ptr` type($base_addr) attr-dict
  }];

  // FIXME: add verifier.
  // Check whether both src/dst pointee's are compatible.
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : CIR_Op<"func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove, Symbol
]> {
  let summary = "Declare or define a function";
  let description = [{

    Similar to `mlir::FuncOp` built-in:
    > Operations within the function cannot implicitly capture values defined
    > outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    > external references must use function arguments or attributes that establish
    > a symbolic connection (e.g. symbols referenced by name via a string
    > attribute like SymbolRefAttr). An external function declaration (used when
    > referring to a function declared in some other module) has no body. While
    > the MLIR textual form provides a nice inline syntax for function arguments,
    > they are internally represented as “block arguments” to the first block in
    > the region.
    >
    > Only dialect attribute names may be specified in the attribute dictionaries
    > for function arguments, results, or the function itself.

    The function linkage information is specified by `linkage`, as defined by
    `GlobalLinkageKind` attribute.

    A compiler builtin function must be marked as `builtin` for further
    processing when lowering from CIR.

    The `coroutine` keyword is used to mark coroutine function, which requires
    at least one `cir.await` instruction to be used in its body.

    The `lambda` translates to a C++ `operator()` that implements a lambda, this
    allow callsites to make certain assumptions about the real function nature
    when writing analysis. The verifier should, but do act on this keyword yet.

    The `no_proto` keyword is used to identify functions that were declared
    without a prototype and, consequently, may contain calls with invalid
    arguments and undefined behavior.

    The `extra_attrs`, which is an aggregate of function-specific attributes is
    required and mandatory to describle additional attributes that are not listed
    above. Though mandatory, the prining of the attribute can be omitted if it is
    empty.

    Example:

    ```mlir
    // External function definitions.
    cir.func @abort()

    // A function with internal linkage.
    cir.func internal @count(%x: i64) -> (i64)
      return %x : i64
    }

    // Linkage information
    cir.func linkonce_odr @some_method(...)

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

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<CIR_FuncType>:$function_type,
                       UnitAttr:$builtin,
                       UnitAttr:$coroutine,
                       UnitAttr:$lambda,
                       UnitAttr:$no_proto,
                       DefaultValuedAttr<GlobalLinkageKind,
                                         "GlobalLinkageKind::ExternalLinkage">:$linkage,
                       ExtraFuncAttr:$extra_attrs,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<FlatSymbolRefAttr>:$aliasee,
                       OptionalAttr<AnyASTFunctionDeclAttr>:$ast);
  let regions = (region AnyRegion:$body);
  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FuncType":$type,
    CArg<"GlobalLinkageKind", "GlobalLinkageKind::ExternalLinkage">:$linkage,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion();

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() {
      if (::llvm::isa<cir::VoidType>(getFunctionType().getReturnType()))
        return {};
      return getFunctionType().getReturnTypes();
    }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or null if
    /// there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getReturnTypes(); }

    /// Hook for OpTrait::FunctionOpInterfaceTrait, called after verifying that
    /// the 'type' attribute is present and checks if it holds a function type.
    /// Ensures getType, getNumFuncArguments, and getNumFuncResults can be
    /// called safely.
    LogicalResult verifyType();

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration();

    // FIXME: should be shared with GlobalOp extra declaration.
    bool isDeclarationForLinker() {
      if (mlir::cir::isAvailableExternallyLinkage(getLinkage()))
        return true;

      return isDeclaration();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

class CIR_CallOp<string mnemonic, list<Trait> extra_traits = []> :
    Op<CIR_Dialect, mnemonic,
       !listconcat(extra_traits,
                   [DeclareOpInterfaceMethods<CIRCallOpInterface>,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>])> {
  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    OperandRange getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      llvm_unreachable("NYI");
    }

    /// Return the callee of this operation
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      if (auto calling =
              (*this)->getAttrOfType<mlir::SymbolRefAttr>(getCalleeAttrName()))
        (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
      setOperand(0, callee.get<mlir::Value>());
    }

    bool isIndirect() { return !getCallee(); }
    mlir::Value getIndirectCall();
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let hasVerifier = 0;

  dag commonArgs = (ins
    OptionalAttr<FlatSymbolRefAttr>:$callee,
    Variadic<CIR_AnyType>:$arg_ops,
    OptionalAttr<ASTCallExprInterface>:$ast
  );
}

def CallOp : CIR_CallOp<"call"> {
  let summary = "call operation";
  let description = [{
    Direct and indirect calls.

    For direct calls, the `call` operation represents a direct call to a
    function that is within the same symbol scope as the call. The operands
    and result types of the call must match the specified function type.
    The callee is encoded as a aymbol reference attribute named "callee".

    For indirect calls, the first `mlir::Operation` operand is the call target.

    Given the way indirect calls are encoded, avoid using `mlir::Operation`
    methods to walk the operands for this operation, instead use the methods
    provided by `CIRCallOpInterface`.
    ``

    Example:

    ```mlir
    // Direct call
    %2 = cir.call @my_add(%0, %1) : (f32, f32) -> f32
     ...
    // Indirect call
    %20 = cir.call %18(%17)
    ```
  }];

  let arguments = commonArgs;
  let results = (outs Variadic<CIR_AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      if (!callee.getFunctionType().isVoid())
        $_state.addTypes(callee.getFunctionType().getReturnType());
    }]>,
    OpBuilder<(ins "Value":$ind_target,
               "FuncType":$fn_type,
               CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(ValueRange{ind_target});
      $_state.addOperands(operands);
      if (!fn_type.isVoid())
        $_state.addTypes(fn_type.getReturnType());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "mlir::Type":$resType,
              CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(resType);
    }]>];
}

//===----------------------------------------------------------------------===//
// TryCallOp
//===----------------------------------------------------------------------===//

def TryCallOp : CIR_CallOp<"try_call"> {
  let summary = "try call operation";
  let description = [{
    Similar to `cir.call`, direct and indirect properties are the same. The
    difference relies in an exception object address operand. It's encoded
    as the first operands or second (for indirect calls).

    Similarly to `cir.call`, avoid using `mlir::Operation` methods to walk the
    operands for this operation, instead use the methods provided by
    `CIRCallOpInterface`.

    Example:

    ```mlir
    cir.try {
      %0 = cir.alloca !cir.ptr<!cir.eh.info>, cir.ptr <!cir.ptr<!cir.eh.info>>
      ...
      %r = cir.try_call %exception(%0) @division(%1, %2)
    } ...
    ```
  }];

  let arguments = !con((ins
    ExceptionInfoPtrPtr:$exceptionInfo
  ), commonArgs);

  let results = (outs Variadic<CIR_AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, "mlir::Value":$exception,
                   CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(ValueRange{exception});
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      if (!callee.getFunctionType().isVoid())
        $_state.addTypes(callee.getFunctionType().getReturnType());
    }]>,
    OpBuilder<(ins "Value":$ind_target, "mlir::Value":$exception,
               "FuncType":$fn_type,
               CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(ValueRange{exception});
      $_state.addOperands(ValueRange{ind_target});
      $_state.addOperands(operands);
      if (!fn_type.isVoid())
        $_state.addTypes(fn_type.getReturnType());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "mlir::Value":$exception,
                   "mlir::Type":$resType, CArg<"ValueRange", "{}">:$operands),
    [{
      $_state.addOperands(ValueRange{exception});
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(resType);
    }]>];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

def AK_Initial : I32EnumAttrCase<"init", 1>;
def AK_User    : I32EnumAttrCase<"user", 2>;
def AK_Final   : I32EnumAttrCase<"final", 3>;

def AwaitKind : I32EnumAttr<
    "AwaitKind",
    "await kind",
    [AK_Initial, AK_User, AK_Final]> {
  let cppNamespace = "::mlir::cir";
}

def AwaitOp : CIR_Op<"await",
       [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
        RecursivelySpeculatable, NoRegionArguments]> {
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

    Breaking this up in regions allow individual scrutiny of conditions
    which might lead to folding some of them out. Lowerings coming out
    of CIR, e.g. LLVM, should use the `suspend` region to track more
    lower level codegen (e.g. intrinsic emission for coro.save/coro.suspend).

    There are also 3 flavors of `cir.await` available:
    - `init`: compiler generated initial suspend via implicit `co_await`.
    - `user`: also known as normal, representing user written co_await's.
    - `final`: compiler generated final suspend via implicit `co_await`.

    From the C++ snippet we get:

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

  let arguments = (ins AwaitKind:$kind);
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
      "mlir::cir::AwaitKind":$kind,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$readyBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$suspendBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$resumeBuilder
      )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

def TryOp : CIR_Op<"try",
       [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
        RecursivelySpeculatable, AutomaticAllocationScope,
        NoRegionArguments]> {
  let summary = "";
  let description = [{
    ```mlir
    TBD
    ```

    Note that variables declared inside a `try {}` in C++ will
    have their allocas places in the surrounding (parent) scope.
  }];

  let regions = (region SizedRegion<1>:$body);
  let results = (outs ExceptionInfoPtr:$result);

  let assemblyFormat = [{
    $body `:` functional-type(operands, results) attr-dict
  }];

  // Everything already covered elsewhere.
  let hasVerifier = 0;
  let builders = [
    OpBuilder<(ins
              "function_ref<void(OpBuilder &, Type &, Location)>":$tryBuilder)>,
  ];
}

//===----------------------------------------------------------------------===//
// CatchOp
//===----------------------------------------------------------------------===//

// Represents the unwind region where unwind continues or
// the program std::terminate's.
def CatchUnwind : CIRUnitAttr<"CatchUnwind", "unwind"> {
  let storageType = [{ CatchUnwind }];
}

// Represents the catch_all region.
def CatchAllAttr : CIRUnitAttr<"CatchAll", "all"> {
  let storageType = [{ CatchAllAttr }];
}

def CatchOp : CIR_Op<"catch",
      [SameVariadicOperandSize,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, NoRegionArguments]> {
  let summary = "Catch operation";
  let description = [{
  }];

  let arguments = (ins CIR_AnyType:$exception_info,
                       OptionalAttr<ArrayAttr>:$catchers);
  let regions = (region VariadicRegion<AnyRegion>:$regions);

  // Already verified elsewhere
  let hasVerifier = 0;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$exception_info,
      "function_ref<void(OpBuilder &, Location, OperationState &)>"
      :$catchBuilder)>
  ];

  let assemblyFormat = [{
    `(`
      $exception_info `:` type($exception_info) `,`
      custom<CatchOp>($regions, $catchers)
    `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CatchParamOp
//===----------------------------------------------------------------------===//

def CatchParamOp : CIR_Op<"catch_param"> {
  let summary = "Materialize the catch clause formal parameter";
  let description = [{
    The `cir.catch_param` binds to a the C/C++ catch clause param and allow
    it to be materialized. This operantion grabs the param by looking into
    a exception info `!cir.eh_info` argument.

    Example:
    ```mlir
    // TBD
    ```
  }];

  let arguments = (ins ExceptionInfoPtr:$exception_info);
  let results = (outs CIR_AnyType:$param);
  let assemblyFormat = [{
    `(` $exception_info `)` `->` qualified(type($param)) attr-dict
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def CopyOp : CIR_Op<"copy", [SameTypeOperands]> {
  let arguments = (ins Arg<CIR_PointerType, "", [MemWrite]>:$dst,
                       Arg<CIR_PointerType, "", [MemRead]>:$src);
  let summary = "Copies contents from a CIR pointer to another";
  let description = [{
    Given two CIR pointers, `src` and `dst`, `cir.copy` will copy the memory
    pointed by `src` to the memory pointed by `dst`.

    The amount of bytes copied is inferred from the pointee type. Naturally,
    the pointee type of both `src` and `dst` must match and must implement
    the `DataLayoutTypeInterface`.

    Examples:

    ```mlir
      // Copying contents from one struct to another:
      cir.copy %0 to %1 : !cir.ptr<!struct_ty>
    ```
  }];

  let assemblyFormat = "$src `to` $dst attr-dict `:` qualified(type($dst))";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the pointer type being copied.
    mlir::cir::PointerType getType() { return getSrc().getType(); }

    /// Returns the number of bytes to be copied.
    unsigned getLength() {
      return DataLayout::closest(*this).getTypeSize(getType().getPointee());
    }
  }];
}

//===----------------------------------------------------------------------===//
// MemCpyOp
//===----------------------------------------------------------------------===//

def MemCpyOp : CIR_Op<"libc.memcpy"> {
  let arguments = (ins Arg<CIR_PointerType, "", [MemWrite]>:$dst,
                       Arg<CIR_PointerType, "", [MemRead]>:$src,
                       CIR_IntType:$len);
  let summary = "Equivalent to libc's `memcpy`";
  let description = [{
    Given two CIR pointers, `src` and `dst`, `cir.libc.memcpy` will copy `len`
    bytes from the memory pointed by `src` to the memory pointed by `dst`.

    While `cir.copy` is meant to be used for implicit copies in the code where
    the length of the copy is known, `cir.memcpy` copies only from and to void
    pointers, requiring the copy length to be passed as an argument.

    Examples:

    ```mlir
      // Copying 2 bytes from one array to a struct:
      %2 = cir.const(#cir.int<2> : !u32i) : !u32i
      cir.libc.memcpy %2 bytes from %arr to %struct : !cir.ptr<!arr> -> !cir.ptr<!struct>
    ```
  }];

  let assemblyFormat = [{
    $len `bytes` `from` $src `to` $dst attr-dict
    `:` type($len) `` `,` qualified(type($src)) `->` qualified(type($dst))
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the data source pointer type.
    mlir::cir::PointerType getSrcTy() { return getSrc().getType(); }

    /// Returns the data destination pointer type.
    mlir::cir::PointerType getDstTy() { return getDst().getType(); }

    /// Returns the byte length type.
    mlir::cir::IntType getLenTy() { return getLen().getType(); }
  }];
}

//===----------------------------------------------------------------------===//
// MemChrOp
//===----------------------------------------------------------------------===//

def MemChrOp : CIR_Op<"libc.memchr"> {
  // TODO: instead of using UInt64 for len, we could make it constrained on
  // size_t (64 or 32) and have a builder that does the right job.
  let arguments = (ins Arg<VoidPtr, "", [MemRead]>:$src,
                       SInt32:$pattern,
                       UInt64:$len);
  let summary = "libc's `memchr`";
  let results = (outs Res<VoidPtr, "">:$result);

  let description = [{
    Search for `pattern` in data range from `src` to `src` + `len`.
    provides a bound to the search in `src`. `result` is a pointer to found
    `pattern` or a null pointer.

    Examples:

    ```mlir
    %p = cir.libc.memchr(%src, %pattern, %len) -> !cir.ptr<!void>
    ```
  }];

  let assemblyFormat = [{
    `(`
      $src `,` $pattern `,` $len `)` attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// StdFindOp
//===----------------------------------------------------------------------===//

def StdFindOp : CIR_Op<"std.find", [SameFirstSecondOperandAndResultType]> {
  let arguments = (ins FlatSymbolRefAttr:$original_fn,
                       CIR_AnyType:$first,
                       CIR_AnyType:$last,
                       CIR_AnyType:$pattern);
  let summary = "std:find()";
  let results = (outs CIR_AnyType:$result);

  let description = [{
    Search for `pattern` in data range from `first` to `last`. This currently
    maps to only one form of `std::find`. The `original_fn` operand tracks the
    mangled named that can be used when lowering to a `cir.call`.

    Example:

    ```mlir
    ...
    %result = cir.std.find(@original_fn,
                           %first : !T, %last : !T, %pattern : !P) -> !T
    ```
  }];

  let assemblyFormat = [{
    `(`
      $original_fn
      `,` $first `:` type($first)
      `,` $last `:` type($last)
      `,` $pattern `:` type($pattern)
    `)` `->` type($result) attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// IterBegin/End
//===----------------------------------------------------------------------===//

def IterBeginOp : CIR_Op<"iterator_begin"> {
  let arguments = (ins FlatSymbolRefAttr:$original_fn, CIR_AnyType:$container);
  let summary = "Returns an iterator to the first element of a container";
  let results = (outs CIR_AnyType:$result);
  let assemblyFormat = [{
    `(`
      $original_fn `,` $container `:` type($container)
    `)` `->` type($result) attr-dict
  }];
  let hasVerifier = 0;
}

def IterEndOp : CIR_Op<"iterator_end"> {
  let arguments = (ins FlatSymbolRefAttr:$original_fn, CIR_AnyType:$container);
  let summary = "Returns an iterator to the element following the last element"
                " of a container";
  let results = (outs CIR_AnyType:$result);
  let assemblyFormat = [{
    `(`
      $original_fn `,` $container `:` type($container)
    `)` `->` type($result) attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Floating Point Ops
//===----------------------------------------------------------------------===//

class UnaryFPToFPBuiltinOp<string mnemonic>
    : CIR_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins CIR_AnyFloat:$src);
  let results = (outs CIR_AnyFloat:$result);
  let summary = "libc builtin equivalent ignoring "
                "floating point exceptions and errno";
  let assemblyFormat = "$src `:` type($src) attr-dict";
}

def CeilOp : UnaryFPToFPBuiltinOp<"ceil">;
def CosOp : UnaryFPToFPBuiltinOp<"cos">;
def ExpOp : UnaryFPToFPBuiltinOp<"exp">;
def Exp2Op : UnaryFPToFPBuiltinOp<"exp2">;
def FloorOp : UnaryFPToFPBuiltinOp<"floor">;
def FAbsOp : UnaryFPToFPBuiltinOp<"fabs">;
def LogOp : UnaryFPToFPBuiltinOp<"log">;
def Log10Op : UnaryFPToFPBuiltinOp<"log10">;
def Log2Op : UnaryFPToFPBuiltinOp<"log2">;
def NearbyintOp : UnaryFPToFPBuiltinOp<"nearbyint">;
def RintOp : UnaryFPToFPBuiltinOp<"rint">;
def RoundOp : UnaryFPToFPBuiltinOp<"round">;
def SinOp : UnaryFPToFPBuiltinOp<"sin">;
def SqrtOp : UnaryFPToFPBuiltinOp<"sqrt">;
def TruncOp : UnaryFPToFPBuiltinOp<"trunc">;

//===----------------------------------------------------------------------===//
// Branch Probability Operations
//===----------------------------------------------------------------------===//

def ExpectOp : CIR_Op<"expect",
  [Pure, AllTypesMatch<["result", "val", "expected"]>]> {
  let summary =
    "Compute whether expression is likely to evaluate to a specified value";
  let description = [{
    Provides __builtin_expect functionality in Clang IR.

    If $prob is not specified, then behaviour is same as __builtin_expect.
    If specified, then behaviour is same as __builtin_expect_with_probability,
    where probability = $prob.
  }];

  let arguments = (ins CIR_IntType:$val,
                       CIR_IntType:$expected,
                       OptionalAttr<F64Attr>:$prob);
  let results = (outs CIR_IntType:$result);
  let assemblyFormat = [{
    `(` $val`,` $expected (`,` $prob^)? `)` `:` type($val) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Variadic Operations
//===----------------------------------------------------------------------===//

def VAStartOp : CIR_Op<"va.start">, Arguments<(ins CIR_PointerType:$arg_list)> {
  let summary = "Starts a variable argument list";
  let assemblyFormat = "$arg_list attr-dict `:` type(operands)";
  let hasVerifier = 0;
}

def VAEndOp : CIR_Op<"va.end">, Arguments<(ins CIR_PointerType:$arg_list)> {
  let summary = "Ends a variable argument list";
  let assemblyFormat = "$arg_list attr-dict `:` type(operands)";
  let hasVerifier = 0;
}

def VACopyOp : CIR_Op<"va.copy">,
               Arguments<(ins CIR_PointerType:$dst_list,
                              CIR_PointerType:$src_list)> {
  let summary = "Copies a variable argument list";
  let assemblyFormat = "$src_list `to` $dst_list attr-dict `:` type(operands)";
  let hasVerifier = 0;
}

def VAArgOp : CIR_Op<"va.arg">,
              Results<(outs CIR_AnyType:$result)>,
              Arguments<(ins CIR_PointerType:$arg_list)> {
  let summary = "Fetches next variadic element as a given type";
  let assemblyFormat = "$arg_list attr-dict `:` functional-type(operands, $result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AllocException
//===----------------------------------------------------------------------===//

def AllocException : CIR_Op<"alloc_exception", [
  AllocaTypesMatchWith<"'allocType' matches pointee type of 'addr'",
                 "addr", "allocType",
                 "$_self.cast<PointerType>().getPointee()">]> {
  let summary = "Defines a scope-local variable";
  let description = [{
    Implements a slightly higher level __cxa_allocate_exception:

    `void *__cxa_allocate_exception(size_t thrown_size);`

    If operation fails, program terminates, not throw.

    Example:

    ```mlir
    // if (b == 0) {
    //   ...
    //   throw "...";
    cir.if %10 {
        %11 = cir.alloc_exception(!cir.ptr<!u8i>) -> <!cir.ptr<!u8i>>
        ... // store exception content into %11
        cir.throw(%11 : !cir.ptr<!cir.ptr<!u8i>>, ...
    ```
  }];

  let arguments = (ins TypeAttr:$allocType);
  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<DefaultResource>]>:$addr);

  let assemblyFormat = [{
    `(` $allocType `)` `->` type($addr) attr-dict
  }];

  // Constraints verified elsewhere.
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// ThrowOp
//===----------------------------------------------------------------------===//

def ThrowOp : CIR_Op<"throw", [
                     ParentOneOf<["FuncOp", "ScopeOp", "IfOp", "SwitchOp",
                                  "DoWhileOp", "WhileOp", "ForOp"]>,
                     Terminator]> {
  let summary = "(Re)Throws an exception";
  let description = [{
    Very similar to __cxa_throw:

    ```
    void __cxa_throw(void *thrown_exception, std::type_info *tinfo,
                     void (*dest) (void *));
    ```

    The absense of arguments for `cir.throw` means it rethrows.

    For the no-rethrow version, it must have at least two operands, the RTTI
    information, a pointer to the exception object (likely allocated via
    `cir.cxa.allocate_exception`) and finally an optional dtor, which might
    run as part of this operation.

    ```mlir
      // if (b == 0)
      //   throw "Division by zero condition!";
      cir.if %10 {
        %11 = cir.alloc_exception(!cir.ptr<!u8i>) -> <!cir.ptr<!u8i>>
        ...
        cir.store %13, %11 : // Store string addr for "Division by zero condition!"
        cir.throw(%11 : !cir.ptr<!cir.ptr<!u8i>>, @"typeinfo for char const*")
    ```
  }];

  let arguments = (ins Optional<CIR_AnyType>:$exception_ptr,
                       OptionalAttr<FlatSymbolRefAttr>:$type_info,
                       OptionalAttr<FlatSymbolRefAttr>:$dtor);

  let assemblyFormat = [{
    `(`
      ($exception_ptr^ `:` type($exception_ptr))?
      (`,` $type_info^)?
      (`,` $dtor^)?
    `)` attr-dict
  }];

  let extraClassDeclaration = [{
    bool rethrows() { return getNumOperands() == 0; }
  }];

  let hasVerifier = 1;
}

def StackSaveOp : CIR_Op<"stack_save"> {
  let summary = "remembers the current state of the function stack";
  let description = [{
    Remembers the current state of the function stack. Returns a pointer
    that later can be passed into cir.stack_restore.
    Useful for implementing language features like variable length arrays.

    ```mlir
    %0 = cir.stack_save : <!u8i>
    ```

  }];

  let results = (outs CIR_PointerType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def StackRestoreOp : CIR_Op<"stack_restore"> {
  let summary = "restores the state of the function stack";
  let description = [{
    Restore the state of the function stack to the state it was
    in when the corresponding cir.stack_save executed.
    Useful for implementing language features like variable length arrays.

    ```mlir
    %0 = cir.alloca !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>, ["saved_stack"] {alignment = 8 : i64}
    %1 = cir.stack_save : <!u8i>
    cir.store %1, %0 : !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>
    %2 = cir.load %0 : cir.ptr <!cir.ptr<!u8i>>, !cir.ptr<!u8i>
    cir.stack_restore %2 : !cir.ptr<!u8i>
    ```
    }];

  let arguments = (ins CIR_PointerType:$ptr);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr))";
}

def AsmATT : I32EnumAttrCase<"x86_att", 0>;
def AsmIntel : I32EnumAttrCase<"x86_intel", 1>;

def AsmFlavor : I32EnumAttr<
  "AsmFlavor",
  "ATT or Intel",
  [AsmATT, AsmIntel]> {
  let cppNamespace = "::mlir::cir";
}

def CIR_InlineAsmOp : CIR_Op<"asm", [RecursiveMemoryEffects]> {
  let description = [{
    The `cir.asm` operation represents C/C++ asm inline.

    CIR constraints strings follow barelly the same rules that are established 
    for the C level assembler constraints with several differences caused by 
    clang::AsmStmt processing. 

    Thus, numbers that appears in the constraint string may also refer to:
    - the output variable index referenced by the input operands.
    - the index of early-clobber operand

    Operand attributes is a storage of attributes, where each element corresponds
    to the operand with the same index. The first index relates to the operation
    result.

    Example:
    ```C++
    __asm__("foo" : : : );
    __asm__("bar $42 %[val]" : [val] "=r" (x), "+&r"(x));
    __asm__("baz $42 %[val]" : [val] "=r" (x), "+&r"(x) : "[val]"(y));
    ```
    
    ```mlir
    %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] 
    %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["y", init]
    ... 
    %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
    %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
    cir.asm(x86_att, {"foo" "~{dirflag},~{fpsr},~{flags}"}) side_effects : () -> ()
    cir.asm(x86_att, {"bar $$42 $0" "=r,=&r,1,~{dirflag},~{fpsr},~{flags}"}) %2 : (!s32i) -> ()
    cir.asm(x86_att, {"baz $$42 $0" "=r,=&r,0,1,~{dirflag},~{fpsr},~{flags}"}) %3, %2 : (!s32i, !s32i) -> ()
    ```
  }];

  let results = (outs Optional<CIR_AnyType>:$res);

  let arguments = (
    ins Variadic<AnyType>:$operands,
        StrAttr:$asm_string,
        StrAttr:$constraints,
        UnitAttr:$side_effects,
        AsmFlavor:$asm_flavor,
        OptionalAttr<ArrayAttr>:$operand_attrs);

  let assemblyFormat = [{
    `(`
    $asm_flavor`,`
    `{` $asm_string $constraints `}`
    `)`
    (`operand_attrs` `=` $operand_attrs^)?
    (`side_effects` $side_effects^)?
    attr-dict
    operands `:` functional-type(operands, results)
   }];

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
// PrefetchOp
//===----------------------------------------------------------------------===//

def PrefetchOp : CIR_Op<"prefetch"> {
  let summary = "prefetch operation";
  let description = [{
    The `cir.prefetch` op prefetches data from the memmory address.

    ```mlir
    cir.prefetch(%0 : !cir.ptr<!void>) locality(1) write
    ```

    This opcode has the three attributes:
    1. The $locality is a temporal locality specifier
    ranging from (0) - no locality, to (3) - extremely local keep in cache.
    2. The $isWrite is the specifier determining if the prefetch is prepaired
    for a 'read' or 'write'.
    If $isWrite doesn't specified it means that prefetch is prepared for 'read'.
  }];

  let arguments = (
    ins VoidPtr:$addr,
        ConfinedAttr<I32Attr, [IntMinValue<0>,
          IntMaxValue<3>]>:$locality,
        UnitAttr:$isWrite);

  let assemblyFormat = [{
    `(` $addr `:` qualified(type($addr)) `)`
        `locality``(` $locality `)`
        (`write` $isWrite^) : (`read`)?
        attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ArrayCtor & ArrayDtor
//===----------------------------------------------------------------------===//

class CIR_ArrayInitDestroy<string mnemonic> : CIR_Op<mnemonic, []> {
  let arguments = (ins Arg<ArrayPtr, "array address",
                           [MemWrite, MemRead]>:$addr);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `(` $addr `:` qualified(type($addr)) `)` $body attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$addr,
      "function_ref<void(OpBuilder &, Location)>":$regionBuilder), [{
        assert(regionBuilder && "builder callback expected");
        OpBuilder::InsertionGuard guard($_builder);
        Region *r = $_state.addRegion();
        $_state.addOperands(ValueRange{addr});
        $_builder.createBlock(r);
        regionBuilder($_builder, $_state.location);
    }]>
  ];
}

def ArrayCtor : CIR_ArrayInitDestroy<"array.ctor"> {
  let summary = "Initialize array elements with C++ constructors";
  let description = [{
    Initialize each array element using the same C++ constructor. This
    operation has one region, with one single block. The block has an
    incoming argument for the current array index to initialize.
  }];
}

def ArrayDtor : CIR_ArrayInitDestroy<"array.dtor"> {
  let summary = "Destroy array elements with C++ dtors";
  let description = [{
    Destroy each array element using the same C++ destructor. This
    operation has one region, with one single block. The block has an
    incoming argument for the current array index to initialize.
  }];
}

//===----------------------------------------------------------------------===//
// IsConstantOp
//===----------------------------------------------------------------------===//

def IsConstantOp : CIR_Op<"is_constant", [Pure]> {
  let description = [{
    Returns `true` if the argument is known to be a compile-time constant
    otherwise returns 'false'.
  }];
  let arguments = (ins CIR_AnyType:$val);
  let results = (outs CIR_BoolType:$result);

  let assemblyFormat = [{
    `(` $val `:` type($val) `)` `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

// Memory order related definitions.
def MemOrderRelaxed : I32EnumAttrCase<"Relaxed", 0, "relaxed">;
def MemOrderConsume : I32EnumAttrCase<"Consume", 1, "consume">;
def MemOrderAcquire : I32EnumAttrCase<"Acquire", 2, "acquire">;
def MemOrderRelease : I32EnumAttrCase<"Release", 3, "release">;
def MemOrderAcqRel : I32EnumAttrCase<"AcquireRelease", 4, "acq_rel">;
def MemOrderSeqCst : I32EnumAttrCase<"SequentiallyConsistent", 5, "seq_cst">;

def MemOrder : I32EnumAttr<
    "MemOrder",
    "Memory order according to C++11 memory model",
    [MemOrderRelaxed, MemOrderConsume, MemOrderAcquire,
     MemOrderRelease, MemOrderAcqRel, MemOrderSeqCst]> {
  let cppNamespace = "::mlir::cir";
}

// Binary opcodes for atomic fetch.
def Atomic_Add  : I32EnumAttrCase<"Add",  0, "add">;
def Atomic_Sub  : I32EnumAttrCase<"Sub",  1, "sub">;
def Atomic_And  : I32EnumAttrCase<"And",  2, "and">;
def Atomic_Xor  : I32EnumAttrCase<"Xor",  3, "xor">;
def Atomic_Or   : I32EnumAttrCase<"Or",   4, "or">;
def Atomic_Nand : I32EnumAttrCase<"Nand", 5, "nand">;

def AtomicFetchKind : I32EnumAttr<
    "AtomicFetchKind",
    "Binary opcode for cir.atomic.fetch.binop and cir.atomic.binop.fetch",
    [Atomic_Add, Atomic_Sub, Atomic_And,
     Atomic_Xor, Atomic_Or, Atomic_Nand]> {
  let cppNamespace = "::mlir::cir";
}

def AtomicBinopFetch : CIR_Op<"atomic.binop_fetch",
                              [Pure, SameSecondOperandAndResultType]> {
  let summary = "Represents the __atomic_binop_fetch builtin";
  let description = [{}];
  let results = (outs CIR_AnyIntOrFloat:$result);
  let arguments = (ins IntOrFPPtr:$ptr, CIR_AnyIntOrFloat:$val,
                       AtomicFetchKind:$binop,
                       Arg<MemOrder, "memory order">:$mem_order,
                       UnitAttr:$is_volatile);

  let assemblyFormat = [{
    `(`
    $binop `,`
    $ptr `:` type($ptr) `,`
    $val `:` type($val) `,`
    $mem_order `)`
    (`volatile` $is_volatile^)?
    `:` type($result) attr-dict
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Operations Lowered Directly to LLVM IR
//
// These operations are hacks to get around missing features in LLVM's dialect.
// Use it sparingly and remove it once the features are added.
//===----------------------------------------------------------------------===//

def ZeroInitConstOp : CIR_Op<"llvmir.zeroinit", [Pure]>,
                      Results<(outs AnyType:$result)> {
  let summary = "Zero initializes a constant value of a given type";
  let description = [{
    This operation circumvents the lack of a zeroinitializer operation in LLVM
    Dialect. It can zeroinitialize any LLVM type.
  }];
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 0;
}

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIROPS
