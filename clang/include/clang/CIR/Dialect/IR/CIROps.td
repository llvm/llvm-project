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

  // The constant operation takes an attribute as the only input.
  let arguments = (ins TypedAttrInterface:$value);

  // The constant operation returns a single value of CIR_AnyType.
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

  let hasVerifier = 0;
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
    backed up by a `cir.ptr` type.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = cir.load %0 : !cir.ptr<i32>, i32
    ```
  }];

  let arguments = (ins Arg<CIR_PointerType, "the address to load from",
                           [MemRead]>:$addr);
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
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
                           [MemWrite]>:$addr);

  let assemblyFormat = [{
    $value `,` $addr attr-dict `:` type($value) `,` qualified(type($addr))
  }];

  // FIXME: add verifier.
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : CIR_Op<"return", [ParentOneOf<["FuncOp", "ScopeOp"]>,
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
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : CIR_Op<"yield", [ReturnLike, Terminator,
                               ParentOneOf<["ScopeOp"]>]> {
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
// GlobalOp
//===----------------------------------------------------------------------===//

// TODO(CIR): For starters, cir.global has only name and type.  The other
// properties of a global variable will be added over time as more of ClangIR
// is upstreamed.

def GlobalOp : CIR_Op<"global"> {
  let summary = "Declare or define a global variable";
  let description = [{
    The `cir.global` operation declares or defines a named global variable.

    The backing memory for the variable is allocated statically and is
    described by the type of the variable.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$sym_type,
                       OptionalAttr<AnyAttr>:$initial_value);

  let assemblyFormat = [{
    $sym_name
    custom<GlobalOpTypeAndInitialValue>($sym_type, $initial_value)
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool isDeclaration() { return !getInitialValue(); }
    bool hasInitializer() { return !isDeclaration(); }
  }];

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins "llvm::StringRef":$sym_name,
                                 "mlir::Type":$sym_type)>];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// TODO(CIR): FuncOp is still a tiny shell of what it will become.  Many more
// properties and attributes will be added as upstreaming continues.

def FuncOp : CIR_Op<"func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove
]> {
  let summary = "Declare or define a function";
  let description = [{
    The `cir.func` operation defines a function, similar to the `mlir::FuncOp`
    built-in.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<CIR_FuncType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins "llvm::StringRef":$sym_name,
                                 "FuncType":$type)>];

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
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
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

#endif // CLANG_CIR_DIALECT_IR_CIROPS_TD
