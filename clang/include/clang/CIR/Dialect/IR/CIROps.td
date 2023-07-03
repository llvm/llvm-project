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

include "mlir/Interfaces/CallInterfaces.td"
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

def CastKind : I32EnumAttr<
    "CastKind",
    "cast kind",
    [CK_IntegralToBoolean, CK_ArrayToPointerDecay, CK_IntegralCast,
     CK_BitCast, CK_FloatingCast, CK_PtrToBoolean, CK_FloatToIntegral,
     CK_IntegralToPointer, CK_PointerToIntegral, CK_FloatToBoolean,
     CK_BooleanToIntegral, CK_IntegralToFloat]> {
  let cppNamespace = "::mlir::cir";
}

def CastOp : CIR_Op<"cast", [Pure]> {
  // FIXME: not all conversions are free of side effects.
  let summary = "Conversion between values of different types";
  let description = [{
    Apply C/C++ usual conversions rules between values. Currently supported kinds:

    - `int_to_bool`
    - `ptr_to_bool`
    - `array_to_ptrdecay`
    - `integral`
    - `bitcast`
    - `floating`
    - `float_to_int`

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

  let arguments = (ins CastKind:$kind, AnyType:$src);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $kind `,` $src `:` type($src) `)`
    `,` type($result) attr-dict
  }];

  // The input and output types should match the cast kind.
  let hasVerifier = 1;
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
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs  `)` `:` type($lhs) `->` type($result) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// PtrStrideOp
//===----------------------------------------------------------------------===//

def SameFirstOperandAndResultType :
  NativeOpTrait<"SameFirstOperandAndResultType">;

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

  let arguments = (ins AnyType:$base, CIR_IntType:$stride);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $base `:` type($base) `,` $stride `:` qualified(type($stride)) `)`
    `,` type($result) attr-dict
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

  // The constant operation returns a single value of AnyType.
  let results = (outs AnyType:$res);

  let assemblyFormat = [{
    `(` custom<ConstantValue>($value) `)` attr-dict `:` type($res)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
     bool isNullPtr() {
       return getValue().isa<mlir::cir::NullAttr>();
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
    TypeAttr:$allocaType,
    StrAttr:$name,
    UnitAttr:$init,
    ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment,
    OptionalAttr<ASTVarDeclAttr>:$ast
  );

  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$addr, "Type":$allocaType,
                   "StringRef":$name,
                   "IntegerAttr":$alignment)>
  ];

  let extraClassDeclaration = [{
    // Whether the alloca input type is a pointer.
    bool isPointerType() { return getAllocaType().isa<::mlir::cir::PointerType>(); }
  }];

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat = [{
    $allocaType `,` `cir.ptr` type($addr) `,`
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
    a pointer.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = cir.load %0 : !cir.ptr<i32>, i32

    // Load address from memory at address %0. %3 is used by at least one
    // operation that dereferences a pointer.
    %3 = cir.load deref %0 : cir.ptr <!cir.ptr<i32>>
    ```
  }];

  let arguments = (ins Arg<CIR_PointerType, "the address to load from",
                           [MemRead]>:$addr, UnitAttr:$isDeref);
  let results = (outs AnyType:$result);

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat = [{
    (`deref` $isDeref^)? $addr `:` `cir.ptr` type($addr) `,`
                                   type($result) attr-dict
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
    in the second operand.

    Example:

    ```mlir
    // Store a function argument to local storage, address in %0.
    cir.store %arg0, %0 : i32, !cir.ptr<i32>
    ```
  }];

  let arguments = (ins AnyType:$value,
                       Arg<CIR_PointerType, "the address to store the value",
                           [MemWrite]>:$addr);

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
  let assemblyFormat =
      "$value `,` $addr attr-dict `:` type($value) `,` `cir.ptr` type($addr)";

  // FIXME: add verifier.
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : CIR_Op<"return", [HasParent<"FuncOp, ScopeOp, IfOp, SwitchOp, LoopOp">,
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
  let arguments = (ins Variadic<AnyType>:$input);

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
  let results = (outs Optional<AnyType>:$result);

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
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOpKind_BK : I32EnumAttrCase<"Break", 1, "break">;
def YieldOpKind_FT : I32EnumAttrCase<"Fallthrough", 2, "fallthrough">;
def YieldOpKind_CE : I32EnumAttrCase<"Continue", 3, "continue">;
def YieldOpKind_NS : I32EnumAttrCase<"NoSuspend", 4, "nosuspend">;

def YieldOpKind : I32EnumAttr<
    "YieldOpKind",
    "yield kind",
    [YieldOpKind_BK, YieldOpKind_FT, YieldOpKind_CE, YieldOpKind_NS]> {
  let cppNamespace = "::mlir::cir";
}

def YieldOp : CIR_Op<"yield", [ReturnLike, Terminator,
    ParentOneOf<["IfOp", "ScopeOp", "SwitchOp", "LoopOp", "AwaitOp",
                 "TernaryOp"]>]> {
  let summary = "Terminate CIR regions";
  let description = [{
    The `cir.yield` operation terminates regions on different CIR operations:
    `cir.if`, `cir.scope`, `cir.switch`, `cir.loop`, `cir.await` and `cir.ternary`.

    Might yield an SSA value and the semantics of how the values are yielded is
    defined by the parent operation.

    Optionally, `cir.yield` can be annotated with extra kind specifiers:
    - `break`: breaking out of the innermost `cir.switch` / `cir.loop` semantics,
    cannot be used if not dominated by these parent operations.
    - `fallthrough`: execution falls to the next region in `cir.switch` case list.
    Only available inside `cir.switch` regions.
    - `continue`: only allowed under `cir.loop`, continue execution to the next
    loop step.
    - `nosuspend`: specific to the `ready` region inside `cir.await` op, it makes
    control-flow to be transfered back to the parent, preventing suspension.

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
        cir.yield fallthrough
      }, ...
    ]

    cir.loop (cond : {...}, step : {...}) {
      ...
      cir.yield continue
    }

    cir.await(init, ready : {
      // Call std::suspend_always::await_ready
      %18 = cir.call @_ZNSt14suspend_always11await_readyEv(...)
      cir.if %18 {
        // yields back to the parent.
        cir.yield nosuspend
      }
      cir.yield // control-flow to the next region for suspension.
    }, ...)

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

  let arguments = (ins OptionalAttr<YieldOpKind>:$kind,
                       Variadic<AnyType>:$args);
  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
    OpBuilder<(ins "YieldOpKind":$kind), [{
      mlir::cir::YieldOpKindAttr kattr = mlir::cir::YieldOpKindAttr::get(
        $_builder.getContext(), kind);
      $_state.addAttribute(getKindAttrName($_state.name), kattr);
    }]>,
    OpBuilder<(ins "ValueRange":$results), [{
      $_state.addOperands(results);
    }]>
  ];

  let assemblyFormat = [{
    ($kind^)? ($args^ `:` type($args))? attr-dict
  }];

  let extraClassDeclaration = [{
    // None of the below
    bool isPlain() {
      return !getKind();
    }
    bool isFallthrough() {
      return !isPlain() && *getKind() == YieldOpKind::Fallthrough;
    }
    bool isBreak() {
      return !isPlain() && *getKind() == YieldOpKind::Break;
    }
    bool isContinue() {
      return !isPlain() && *getKind() == YieldOpKind::Continue;
    }
    bool isNoSuspend() {
      return !isPlain() && *getKind() == YieldOpKind::NoSuspend;
    }
  }];

  let hasVerifier = 1;
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

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$scopeRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

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

    Note for inc and dec: the operation corresponds only to the
    addition/subtraction, its input is expect to come from a load
    and the result to be used by a corresponding store.

    It requires one input operand and has one result, both types
    should be the same.

    ```mlir
    %7 = cir.unary(inc, %1) : i32 -> i32
    %8 = cir.unary(dec, %2) : i32 -> i32
    ```
  }];

  let results = (outs AnyType:$result);
  let arguments = (ins Arg<UnaryOpKind, "unary op kind">:$kind, Arg<AnyType>:$input);

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
def BinOpKind_Shl : I32EnumAttrCase<"Shl", 6, "shl">;
def BinOpKind_Shr : I32EnumAttrCase<"Shr", 7, "shr">;
def BinOpKind_And : I32EnumAttrCase<"And", 8, "and">;
def BinOpKind_Xor : I32EnumAttrCase<"Xor", 9, "xor">;
def BinOpKind_Or  : I32EnumAttrCase<"Or", 10, "or">;

def BinOpKind : I32EnumAttr<
    "BinOpKind",
    "binary operation (arith and logic) kind",
    [BinOpKind_Mul, BinOpKind_Div, BinOpKind_Rem,
     BinOpKind_Add, BinOpKind_Sub, BinOpKind_Shl,
     BinOpKind_Shr, BinOpKind_And, BinOpKind_Xor,
     BinOpKind_Or]> {
  let cppNamespace = "::mlir::cir";
}

// FIXME: Pure won't work when we add overloading.
def BinOp : CIR_Op<"binop", [Pure,
  SameTypeOperands, SameOperandsAndResultType]> {

  let summary = "Binary operations (arith and logic)";
  let description = [{
    cir.binop performs the binary operation according to
    the specified opcode kind: [mul, div, rem, add, sub, shl,
    shr, and, xor, or].

    It requires two input operands and has one result, all types
    should be the same.

    ```mlir
    %7 = binop(add, %1, %2) : i32
    %7 = binop(mul, %1, %2) : i8
    ```
  }];

  // TODO: get more accurate than AnyType
  let results = (outs AnyType:$result);
  let arguments = (ins Arg<BinOpKind, "binop kind">:$kind,
                       AnyType:$lhs, AnyType:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) attr-dict
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

  // TODO: get more accurate than AnyType
  let results = (outs AnyType:$result);
  let arguments = (ins Arg<CmpOpKind, "cmp kind">:$kind,
                       AnyType:$lhs, AnyType:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) `,` type($result) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
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

  let arguments = (ins Variadic<AnyType>:$destOperands);
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
                       Variadic<AnyType>:$destOperandsTrue,
                       Variadic<AnyType>:$destOperandsFalse);
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
// LoopOp
//===----------------------------------------------------------------------===//

def LoopOpKind_For : I32EnumAttrCase<"For", 1, "for">;
def LoopOpKind_While : I32EnumAttrCase<"While", 2, "while">;
def LoopOpKind_DoWhile : I32EnumAttrCase<"DoWhile", 3, "dowhile">;

def LoopOpKind : I32EnumAttr<
    "LoopOpKind",
    "Loop kind",
    [LoopOpKind_For, LoopOpKind_While, LoopOpKind_DoWhile]> {
  let cppNamespace = "::mlir::cir";
}

def LoopOp : CIR_Op<"loop",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, NoRegionArguments]> {
  let summary = "Loop";
  let description = [{
    `cir.loop` represents C/C++ loop forms. It defines 3 blocks:
    - `cond`: region can contain multiple blocks, terminated by regular
    `cir.yield` when control should yield back to the parent, and
    `cir.yield continue` when execution continues to another region.
    The region destination depends on the loop form specified.
    - `step`: region with one block, containing code to compute the
    loop step, must be terminated with `cir.yield`.
    - `body`: region for the loop's body, can contain an arbitrary
    number of blocks.

    The loop form: `for`, `while` and `dowhile` must also be specified and
    each implies the loop regions execution order.

    ```mlir
      // while (true) {
      //  i = i + 1;
      // }
      cir.loop while(cond :  {
        cir.yield continue
      }, step :  {
        cir.yield
      })  {
        %3 = cir.load %1 : cir.ptr <i32>, i32
        %4 = cir.const(1 : i32) : i32
        %5 = cir.binop(add, %3, %4) : i32
        cir.store %5, %1 : i32, cir.ptr <i32>
        cir.yield
      }
    ```
  }];

  let arguments = (ins Arg<LoopOpKind, "loop kind">:$kind);
  let regions = (region AnyRegion:$cond, AnyRegion:$body,
                        SizedRegion<1>:$step);

  let assemblyFormat = [{
    $kind
    `(`
    `cond` `:` $cond `,`
    `step` `:` $step
    `)`
    $body
    attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "cir::LoopOpKind":$kind,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$condBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$bodyBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$stepBuilder
      )>
  ];

  let hasVerifier = 1;
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


def GlobalOp : CIR_Op<"global", [Symbol]> {
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
                       OptionalAttr<I64Attr>:$alignment);

  let assemblyFormat = [{
       ($sym_visibility^)?
       (`constant` $constant^)?
       $linkage
       $sym_name
       custom<GlobalOpTypeAndInitialValue>($sym_type, $initial_value)
       attr-dict
  }];

  let extraClassDeclaration = [{
    bool isDeclaration() {
      return !getInitialValue();
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
            "cir::GlobalLinkageKind::ExternalLinkage">:$linkage
      )>
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
                       Optional<AnyType>:$sym_addr,
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
// StructElementAddr
//===----------------------------------------------------------------------===//

// FIXME: rename this among the lines of GetGlobalOp.
def StructElementAddr : CIR_Op<"struct_element_addr"> {
  let summary = "Get the address of a member of a struct";
  let description = [{
    The `cir.struct_element_addr` operaration gets the address of a particular
    named member from the input struct.

    It expects a pointer to the base struct as well as the name of the member
    and its field index.

    Example:
    ```mlir
    !ty_22struct2EBar22 = type !cir.struct<"struct.Bar", i32, i8>
    ...
    %0 = cir.alloca !ty_22struct2EBar22, cir.ptr <!ty_22struct2EBar22>
    ...
    %1 = cir.struct_element_addr %0, "Bar.a"
    %2 = cir.load %1 : cir.ptr <int>, int
    ...
    ```
  }];

  let arguments = (ins
    Arg<CIR_PointerType, "the address to load from", [MemRead]>:$struct_addr,
    StrAttr:$member_name,
    IndexAttr:$member_index);

  let results = (outs Res<CIR_PointerType, "">:$result);

  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$value, "llvm::StringRef":$name,
              "unsigned":$index),
    [{
      mlir::APInt fieldIdx(64, index);
      build($_builder, $_state, type, value, name, fieldIdx);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the index of the struct member being accessed.
    uint64_t getIndex() { return getMemberIndex().getZExtValue(); }
  }];

  // FIXME: add verifier.
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
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<FlatSymbolRefAttr>:$aliasee,
                       OptionalAttr<ASTFunctionDeclAttr>:$ast);
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
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : CIR_Op<"call",
    [DeclareOpInterfaceMethods<CallOpInterface>, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a function that is within
    the same symbol scope as the call. The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    // Direct call
    %2 = cir.call @my_add(%0, %1) : (f32, f32) -> f32
     ...
    // Indirect call
    %20 = cir.call %18(%17)
    ```
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

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
    }]>];

  let extraClassDeclaration = [{
    mlir::Value getIndirectCallee() {
      assert(!getCallee() && "only works for indirect call");
      return *arg_operand_begin();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let hasVerifier = 0;
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
              Results<(outs AnyType:$result)>,
              Arguments<(ins CIR_PointerType:$arg_list)> {
  let summary = "Fetches next variadic element as a given type";
  let assemblyFormat = "$arg_list attr-dict `:` functional-type(operands, $result)";
  let hasVerifier = 0;
}

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIROPS
