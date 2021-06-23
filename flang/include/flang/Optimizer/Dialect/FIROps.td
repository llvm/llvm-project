//===-- FIROps.td - FIR operation definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the FIR dialect operations
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_FIR_OPS
#define FORTRAN_DIALECT_FIR_OPS

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "flang/Optimizer/Dialect/FIRTypes.td"

// Base class for FIR operations.
// All operations automatically get a prefix of "fir.".
class fir_Op<string mnemonic, list<OpTrait> traits>
  : Op<fir_Dialect, mnemonic, traits>;

// Base class for FIR operations that take a single argument
class fir_SimpleOp<string mnemonic, list<OpTrait> traits>
  : fir_Op<mnemonic, traits> {

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

// Base builder for allocate operations
def fir_AllocateOpBuilder : OpBuilder<(ins
    "mlir::Type":$inType,
    CArg<"mlir::ValueRange", "{}">:$lenParams,
    CArg<"mlir::ValueRange", "{}">:$sizes,
    CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addTypes(getRefTy(inType));
    $_state.addAttribute("in_type", TypeAttr::get(inType));
    $_state.addOperands(sizes);
    $_state.addAttributes(attributes);
  }]>;

def fir_NamedAllocateOpBuilder : OpBuilder<(ins
    "mlir::Type":$inType,
    "llvm::StringRef":$name,
    CArg<"mlir::ValueRange", "{}">:$lenParams,
    CArg<"mlir::ValueRange","{}">:$sizes,
    CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addTypes(getRefTy(inType));
    $_state.addAttribute("in_type", TypeAttr::get(inType));
    if (!name.empty())
      $_state.addAttribute("name", $_builder.getStringAttr(name));
    $_state.addOperands(sizes);
    $_state.addAttributes(attributes);
  }]>;

def fir_OneResultOpBuilder : OpBuilder<(ins
    "mlir::Type":$resultType,
    "mlir::ValueRange":$operands,
    CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes),
  [{
    if (resultType)
      $_state.addTypes(resultType);
    $_state.addOperands(operands);
    $_state.addAttributes(attributes);
  }]>;

// Base class of FIR operations that return 1 result
class fir_OneResultOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, traits>, Results<(outs fir_Type:$res)> {
  let builders = [fir_OneResultOpBuilder];
}

// Base class of FIR operations that have 1 argument and return 1 result
class fir_SimpleOneResultOp<string mnemonic, list<OpTrait> traits = []> :
    fir_SimpleOp<mnemonic, traits> {
  let builders = [fir_OneResultOpBuilder];
}

class fir_TwoBuilders<OpBuilder b1, OpBuilder b2> {
  list<OpBuilder> builders = [b1, b2];
}

class fir_AllocatableBaseOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, traits>, Results<(outs fir_Type:$res)> {
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<BoolAttr>:$target
  );
}

class fir_AllocatableOp<string mnemonic, Resource resource,
      list<OpTrait> traits = []> :
    fir_AllocatableBaseOp<mnemonic,
	!listconcat(traits, [MemoryEffects<[MemAlloc<resource>]>])>,
    fir_TwoBuilders<fir_AllocateOpBuilder, fir_NamedAllocateOpBuilder>,
    Arguments<(ins TypeAttr:$in_type, Variadic<AnyIntegerType>:$args)> {

  let parser = [{
    mlir::Type intype;
    if (parser.parseType(intype))
      return mlir::failure();
    auto &builder = parser.getBuilder();
    result.addAttribute(inType(), mlir::TypeAttr::get(intype));
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> operands;
    llvm::SmallVector<mlir::Type, 8> typeVec;
    bool hasOperands = false;
    if (!parser.parseOptionalLParen()) {
      // parse the LEN params of the derived type. (<params> : <types>)
      if (parser.parseOperandList(operands,
                                  mlir::OpAsmParser::Delimiter::None) ||
          parser.parseColonTypeList(typeVec) ||
          parser.parseRParen())
        return mlir::failure();
      auto lens = builder.getI32IntegerAttr(operands.size());
      result.addAttribute(lenpName(), lens);
      hasOperands = true;
    }
    if (!parser.parseOptionalComma()) {
      // parse size to scale by, vector of n dimensions of type index
      auto opSize = operands.size();
      if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None))
        return mlir::failure();
      for (auto i = opSize, end = operands.size(); i != end; ++i)
        typeVec.push_back(builder.getIndexType());
      hasOperands = true;
    }
    if (hasOperands &&
        parser.resolveOperands(operands, typeVec, parser.getNameLoc(),
                               result.operands))
      return mlir::failure();
    mlir::Type restype = wrapResultType(intype);
    if (!restype) {
      parser.emitError(parser.getNameLoc(), "invalid allocate type: ")
          << intype;
      return mlir::failure();
    }
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.addTypeToList(restype, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << (*this)->getAttr(inType());
    if (hasLenParams()) {
      // print the LEN parameters to a derived type in parens
      p << '(' << getLenParams() << " : " << getLenParams().getTypes() << ')';
    }
    // print the shape of the allocation (if any); all must be index type
    for (auto sh : getShapeOperands()) {
      p << ", ";
      p.printOperand(sh);
    }
    p.printOptionalAttrDict((*this)->getAttrs(), {inType(), lenpName()});
  }];

  string extraAllocClassDeclaration = [{
    static constexpr llvm::StringRef inType() { return "in_type"; }
    static constexpr llvm::StringRef lenpName() { return "len_param_count"; }
    mlir::Type getAllocatedType();

    bool hasLenParams() { return bool{(*this)->getAttr(lenpName())}; }
    bool hasShapeOperands() { return numShapeOperands() > 0; }

    unsigned numLenParams() {
      if (auto val = (*this)->getAttrOfType<mlir::IntegerAttr>(lenpName()))
        return val.getInt();
      return 0;
    }

    operand_range getLenParams() {
      return {operand_begin(), operand_begin() + numLenParams()};
    }

    unsigned numShapeOperands() {
      return operand_end() - operand_begin() + numLenParams();
    }

    operand_range getShapeOperands() {
      return {operand_begin() + numLenParams(), operand_end()};
    }

    static mlir::Type getRefTy(mlir::Type ty);

    /// Get the input type of the allocation
    mlir::Type getInType() {
      return (*this)->getAttrOfType<mlir::TypeAttr>(inType()).getValue();
    }
  }];

  // Verify checks common to all allocation operations
  string allocVerify = [{
    llvm::SmallVector<llvm::StringRef, 8> visited;
    if (verifyInType(getInType(), visited, numShapeOperands()))
      return emitOpError("invalid type for allocation");
    if (verifyRecordLenParams(getInType(), numLenParams()))
      return emitOpError("LEN params do not correspond to type");
  }];
}

//===----------------------------------------------------------------------===//
// Memory SSA operations
//===----------------------------------------------------------------------===//

def fir_AllocaOp :
      fir_AllocatableOp<"alloca", AutomaticAllocationScopeResource> {
  let summary = "allocate storage for a temporary on the stack given a type";
  let description = [{
    This primitive operation is used to allocate an object on the stack.  A
    reference to the object of type `!fir.ref<T>` is returned.  The returned
    object has an undefined/uninitialized state.  The allocation can be given
    an optional name.  The allocation may have a dynamic repetition count
    for allocating a sequence of locations for the specified type.

    ```mlir
      %c = ... : i64
      %x = fir.alloca i32
      %y = fir.alloca !fir.array<8 x i64>
      %z = fir.alloca f32, %c

      %i = ... : i16
      %j = ... : i32
      %w = fir.alloca !fir.type<PT(len1:i16, len2:i32)> (%i, %j : i16, i32)
    ```

    Note that in the case of `%z`, a contiguous block of memory is allocated
    and its size is a runtime multiple of a 32-bit REAL value.

    In the case of `%w`, the arguments `%i` and `%j` are LEN parameters
    (`len1`, `len2`) to the type `PT`.

    Finally, the operation is undefined if the ssa-value `%c` is negative.

    Fortran Semantics:
    There is no language mechanism in Fortran to allocate space on the stack
    like C's `alloca()` function. Therefore fir.alloca is not control-flow
    dependent. However, the lifetime of a stack allocation is often limited to
    a small region and a legal implementation may reuse stack storage in other
    regions when there is no conflict. For example, take the following code
    fragment.

    ```fortran
      CALL foo(1)
      CALL foo(2)
      CALL foo(3)
    ```

    A legal implementation can allocate a stack slot and initialize it with the
    constant `1`, then pass that by reference to foo. Likewise for the second
    and third calls to foo, each stack slot being initialized accordingly. It is
    also a conforming implementation to reuse the same stack slot for all three
    calls, just initializing each in turn. This is possible  as the lifetime of
    the copy of each constant need not exceed that of the CALL statement.
    Indeed, a user would likely expect a good Fortran compiler to perform such
    an optimization.

    Until Fortran 2018, procedures defaulted to non-recursive. A legal
    implementation could therefore convert stack allocations to global
    allocations. Such a conversion effectively adds the SAVE attribute to all
    variables.

    Some temporary entities (large arrays) probably should not be stack
    allocated as stack space can often be limited. A legal implementation can
    convert these large stack allocations to heap allocations regardless of
    whether the procedure is recursive or not.
  }];

  let results = (outs fir_ReferenceType);

  let verifier = allocVerify#[{
    mlir::Type outType = getType();
    if (!outType.isa<fir::ReferenceType>())
      return emitOpError("must be a !fir.ref type");
    if (fir::isa_unknown_size_box(fir::dyn_cast_ptrEleTy(outType)))
      return emitOpError("cannot allocate !fir.box of unknown rank or type");
    return mlir::success();
  }];

  let extraClassDeclaration = extraAllocClassDeclaration#[{
    static mlir::Type wrapResultType(mlir::Type intype);
  }];
}

def fir_LoadOp : fir_OneResultOp<"load"> {
  let summary = "load a value from a memory reference";
  let description = [{
    Load a value from a memory reference into an ssa-value (virtual register).
    Produces an immutable ssa-value of the referent type. A memory reference
    has type `!fir.ref<T>`, `!fir.heap<T>`, or `!fir.ptr<T>`.

    ```mlir
      %a = fir.alloca i32
      %l = fir.load %a : !fir.ref<i32>
    ```

    The ssa-value has an undefined value if the memory reference is undefined
    or null.
  }];

  let arguments = (ins Arg<AnyReferenceLike, "", [MemRead]>:$memref);

  let builders = [OpBuilder<(ins "mlir::Value":$refVal),
    [{
      if (!refVal) {
        mlir::emitError($_state.location, "LoadOp has null argument");
        return;
      }
      auto eleTy = fir::dyn_cast_ptrEleTy(refVal.getType());
      if (!eleTy) {
        mlir::emitError($_state.location, "not a memory reference type");
        return;
      }
      $_state.addOperands(refVal);
      $_state.addTypes(eleTy);
    }]
  >];

  let parser = [{
    mlir::Type type;
    mlir::OpAsmParser::OperandType oper;
    if (parser.parseOperand(oper) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(oper, type, result.operands))
       return mlir::failure();
    mlir::Type eleTy;
    if (getElementOf(eleTy, type) ||
        parser.addTypeToList(eleTy, result.types))
       return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(memref());
    p.printOptionalAttrDict((*this)->getAttrs(), {});
    p << " : " << memref().getType();
  }];

  let extraClassDeclaration = [{
    static mlir::ParseResult getElementOf(mlir::Type &ele, mlir::Type ref);
  }];
}

def fir_StoreOp : fir_Op<"store", []> {
  let summary = "store an SSA-value to a memory location";

  let description = [{
    Store an ssa-value (virtual register) to a memory reference.  The stored
    value must be of the same type as the referent type of the memory
    reference.

    ```mlir
      %v = ... : f64
      %p = ... : !fir.ptr<f64>
      fir.store %v to %p : !fir.ptr<f64>
    ```

    The above store changes the value to which the pointer is pointing and not
    the pointer itself. The operation is undefined if the memory reference,
    `%p`, is undefined or null.
  }];

  let arguments = (ins AnyType:$value,
                   Arg<AnyReferenceLike, "", [MemWrite]>:$memref);

  let parser = [{
    mlir::Type type;
    mlir::OpAsmParser::OperandType oper;
    mlir::OpAsmParser::OperandType store;
    if (parser.parseOperand(oper) ||
        parser.parseKeyword("to") ||
        parser.parseOperand(store) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(oper, elementType(type),
          result.operands) ||
        parser.resolveOperand(store, type, result.operands))
       return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(value());
    p << " to ";
    p.printOperand(memref());
    p.printOptionalAttrDict((*this)->getAttrs(), {});
    p << " : " << memref().getType();
  }];

  let verifier = [{
    if (value().getType() != fir::dyn_cast_ptrEleTy(memref().getType()))
      return emitOpError("store value type must match memory reference type");
    if (fir::isa_unknown_size_box(value().getType()))
      return emitOpError("cannot store !fir.box of unknown rank or type");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static mlir::Type elementType(mlir::Type refType);
  }];
}

def fir_UndefOp : fir_OneResultOp<"undefined", [NoSideEffect]> {
  let summary = "explicit undefined value of some type";
  let description = [{
    Constructs an ssa-value of the specified type with an undefined value.
    This operation is typically created internally by the mem2reg conversion
    pass. An undefined value can be of any type except `!fir.ref<T>`.

    ```mlir
      %a = fir.undefined !fir.array<10 x !fir.type<T>>
    ```

    The example creates an array shaped ssa value. The array is rank 1, extent
    10, and each element has type `!fir.type<T>`.
  }];

  let results = (outs AnyType:$intype);

  let assemblyFormat = "type($intype) attr-dict";

  let verifier = [{
    // allow `undef : ref<T>` since it is a possible from transformations
    return mlir::success();
  }];
}

def fir_ZeroOp : fir_OneResultOp<"zero_bits", [NoSideEffect]> {
  let summary = "explicit polymorphic zero value of some type";
  let description = [{
    Constructs an ssa-value of the specified type with a value of zero for all
    bits.

    ```mlir
      %a = fir.zero_bits !fir.box<!fir.array<10 x !fir.type<T>>>
    ```

    The example creates a value of type box where all bits are zero.
  }];

  let results = (outs AnyType:$intype);

  let assemblyFormat = "type($intype) attr-dict";
}

def fir_AllocMemOp : fir_AllocatableOp<"allocmem", DefaultResource> {
  let summary = "allocate storage on the heap for an object of a given type";

  let description = [{
    Creates a heap memory reference suitable for storing a value of the
    given type, T.  The heap refernce returned has type `!fir.heap<T>`.
    The memory object is in an undefined state.  `allocmem` operations must
    be paired with `freemem` operations to avoid memory leaks.

    ```mlir
      %0 = fir.allocmem !fir.array<10 x f32>
      fir.freemem %0 : !fir.heap<!fir.array<10 x f32>>
    ```
  }];

  let results = (outs fir_HeapType);

  let verifier = allocVerify#[{
    mlir::Type outType = getType();
    if (!outType.dyn_cast<fir::HeapType>())
      return emitOpError("must be a !fir.heap type");
    if (fir::isa_unknown_size_box(fir::dyn_cast_ptrEleTy(outType)))
      return emitOpError("cannot allocate !fir.box of unknown rank or type");
    return mlir::success();
  }];

  let extraClassDeclaration = extraAllocClassDeclaration#[{
    static mlir::Type wrapResultType(mlir::Type intype);
  }];
}

def fir_FreeMemOp : fir_Op<"freemem", [MemoryEffects<[MemFree]>]> {
  let summary = "free a heap object";

  let description = [{
    Deallocates a heap memory reference that was allocated by an `allocmem`.
    The memory object that is deallocated is placed in an undefined state
    after `fir.freemem`.  Optimizations may treat the loading of an object
    in the undefined state as undefined behavior.  This includes aliasing
    references, such as the result of an `fir.embox`.

    ```mlir
      %21 = fir.allocmem !fir.type<ZT(p:i32){field:i32}>
      ...
      fir.freemem %21 : !fir.heap<!fir.type<ZT>>
    ```
  }];

  let arguments = (ins Arg<fir_HeapType, "", [MemFree]>:$heapref);

  let assemblyFormat = "$heapref attr-dict `:` type($heapref)";
}

//===----------------------------------------------------------------------===//
// Terminator operations
//===----------------------------------------------------------------------===//

class fir_SwitchTerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, !listconcat(traits, [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<BranchOpInterface>, Terminator])> {

  let arguments = (ins
    AnyType:$selector,
    Variadic<AnyType>:$compareArgs,
    Variadic<AnyType>:$targetArgs
  );

  let results = (outs);

  let successors = (successor VariadicSuccessor<AnySuccessor>:$targets);

  string extraSwitchClassDeclaration = [{
    using Conditions = mlir::Value;

    static constexpr llvm::StringRef getCasesAttr() { return "case_tags"; }

    // The number of destination conditions that may be tested
    unsigned getNumConditions() {
      return (*this)->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).size();
    }

    // The selector is the value being tested to determine the destination
    mlir::Value getSelector() { return selector(); }
    mlir::Value getSelector(llvm::ArrayRef<mlir::Value> operands) {
      return operands[0];
    }

    // The number of blocks that may be branched to
    unsigned getNumDest() { return (*this)->getNumSuccessors(); }

    llvm::Optional<mlir::OperandRange> getCompareOperands(unsigned cond);
    llvm::Optional<llvm::ArrayRef<mlir::Value>> getCompareOperands(
        llvm::ArrayRef<mlir::Value> operands, unsigned cond);

    llvm::Optional<llvm::ArrayRef<mlir::Value>> getSuccessorOperands(
        llvm::ArrayRef<mlir::Value> operands, unsigned cond);
    using BranchOpInterfaceTrait::getSuccessorOperands;

    // Helper function to deal with Optional operand forms
    void printSuccessorAtIndex(mlir::OpAsmPrinter &p, unsigned i) {
      auto *succ = getSuccessor(i);
      auto ops = getSuccessorOperands(i);
      if (ops.hasValue())
        p.printSuccessorAndUseList(succ, ops.getValue());
      else
        p.printSuccessor(succ);
    }

    unsigned targetOffsetSize();
  }];
}

class fir_IntegralSwitchTerminatorOp<string mnemonic,
    list<OpTrait> traits = []> : fir_SwitchTerminatorOp<mnemonic, traits> {

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::Value":$selector,
    "llvm::ArrayRef<int64_t>":$compareOperands,
    "llvm::ArrayRef<mlir::Block *>":$destinations,
    CArg<"llvm::ArrayRef<mlir::ValueRange>", "{}">:$destOperands,
    CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes),
    [{
      $_state.addOperands(selector);
      llvm::SmallVector<mlir::Attribute, 8> ivalues;
      for (auto iv : compareOperands)
        ivalues.push_back($_builder.getI64IntegerAttr(iv));
      ivalues.push_back($_builder.getUnitAttr());
      $_state.addAttribute(getCasesAttr(), $_builder.getArrayAttr(ivalues));
      const auto count = destinations.size();
      for (auto d : destinations)
        $_state.addSuccessors(d);
      const auto opCount = destOperands.size();
      llvm::SmallVector<int32_t, 8> argOffs;
      int32_t sumArgs = 0;
      for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
        if (i < opCount) {
          $_state.addOperands(destOperands[i]);
          const auto argSz = destOperands[i].size();
          argOffs.push_back(argSz);
          sumArgs += argSz;
        } else {
          argOffs.push_back(0);
        }
      }
      $_state.addAttribute(getOperandSegmentSizeAttr(),
        $_builder.getI32VectorAttr({1, 0, sumArgs}));
      $_state.addAttribute(getTargetOffsetAttr(),
        $_builder.getI32VectorAttr(argOffs));
      $_state.addAttributes(attributes);
    }]
  >];

  let parser = [{
    mlir::OpAsmParser::OperandType selector;
    mlir::Type type;
    if (parseSelector(parser, result, selector, type))
      return mlir::failure();

    llvm::SmallVector<mlir::Attribute, 8> ivalues;
    llvm::SmallVector<mlir::Block *, 8> dests;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 8>, 8> destArgs;
    while (true) {
      mlir::Attribute ivalue; // Integer or Unit
      mlir::Block *dest;
      llvm::SmallVector<mlir::Value, 8> destArg;
      mlir::NamedAttrList temp;
      if (parser.parseAttribute(ivalue, "i", temp) ||
          parser.parseComma() ||
          parser.parseSuccessorAndUseList(dest, destArg))
        return mlir::failure();
      ivalues.push_back(ivalue);
      dests.push_back(dest);
      destArgs.push_back(destArg);
      if (!parser.parseOptionalRSquare())
        break;
      if (parser.parseComma())
        return mlir::failure();
    }
    auto &bld = parser.getBuilder();
    result.addAttribute(getCasesAttr(), bld.getArrayAttr(ivalues));
    llvm::SmallVector<int32_t, 8> argOffs;
    int32_t sumArgs = 0;
    const auto count = dests.size();
    for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
      result.addSuccessors(dests[i]);
      result.addOperands(destArgs[i]);
      auto argSize = destArgs[i].size();
      argOffs.push_back(argSize);
      sumArgs += argSize;
    }
    result.addAttribute(getOperandSegmentSizeAttr(),
                        bld.getI32VectorAttr({1, 0, sumArgs}));
    result.addAttribute(getTargetOffsetAttr(), bld.getI32VectorAttr(argOffs));
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector().getType() << " [";
    auto cases = (*this)->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
    auto count = getNumConditions();
    for (decltype(count) i = 0; i != count; ++i) {
      if (i)
        p << ", ";
      auto &attr = cases[i];
      if (auto intAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>())
        p << intAttr.getValue();
      else
        p.printAttribute(attr);
      p << ", ";
      printSuccessorAtIndex(p, i);
    }
    p << ']';
    p.printOptionalAttrDict((*this)->getAttrs(), {getCasesAttr(), getCompareOffsetAttr(),
        getTargetOffsetAttr(), getOperandSegmentSizeAttr()});
  }];

  let verifier = [{
    if (!(getSelector().getType().isa<mlir::IntegerType>() ||
          getSelector().getType().isa<mlir::IndexType>() ||
          getSelector().getType().isa<fir::IntegerType>()))
      return emitOpError("must be an integer");
    auto cases = (*this)->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
    auto count = getNumDest();
    if (count == 0)
      return emitOpError("must have at least one successor");
    if (getNumConditions() != count)
      return emitOpError("number of cases and targets don't match");
    if (targetOffsetSize() != count)
      return emitOpError("incorrect number of successor operand groups");
    for (decltype(count) i = 0; i != count; ++i) {
      auto &attr = cases[i];
      if (!(attr.isa<mlir::IntegerAttr>() || attr.isa<mlir::UnitAttr>()))
        return emitOpError("invalid case alternative");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = extraSwitchClassDeclaration;
}

def fir_SelectOp : fir_IntegralSwitchTerminatorOp<"select"> {
  let summary = "a multiway branch";

  let description = [{
    A multiway branch terminator with similar semantics to C's `switch`
    statement.  A selector value is matched against a list of constants
    of the same type for a match.  When a match is found, control is
    transferred to the corresponding basic block.  A `select` must have
    at least one basic block with a corresponding `unit` match, and
    that block will be selected when all other conditions fail to match.

    ```mlir
      fir.select %arg:i32 [1, ^bb1(%0 : i32),
                           2, ^bb2(%2,%arg,%arg2 : i32,i32,i32),
                          -3, ^bb3(%arg2,%2 : i32,i32),
                           4, ^bb4(%1 : i32),
                        unit, ^bb5]
    ```
  }];
}

def fir_SelectRankOp : fir_IntegralSwitchTerminatorOp<"select_rank"> {
  let summary = "Fortran's SELECT RANK statement";

  let description = [{
    Similar to `select`, `select_rank` provides a way to express Fortran's
    SELECT RANK construct.  In this case, the rank of the selector value
    is matched against constants of integer type.  The structure is the
    same as `select`, but `select_rank` determines the rank of the selector
    variable at runtime to determine the best match.

    ```mlir
      fir.select_rank %arg:i32 [1, ^bb1(%0 : i32),
                                2, ^bb2(%2,%arg,%arg2 : i32,i32,i32),
                                3, ^bb3(%arg2,%2 : i32,i32),
                               -1, ^bb4(%1 : i32),
                             unit, ^bb5]
    ```
  }];
}

def fir_SelectCaseOp : fir_SwitchTerminatorOp<"select_case"> {
  let summary = "Fortran's SELECT CASE statement";

  let description = [{
    Similar to `select`, `select_case` provides a way to express Fortran's
    SELECT CASE construct.  In this case, the selector value is matched
    against variables (not just constants) and ranges.  The structure is
    the same as `select`, but `select_case` allows for the expression of
    more complex match conditions.

    ```mlir
      fir.select_case %arg : i32 [
            #fir.point, %0, ^bb1(%0 : i32),
            #fir.lower, %1, ^bb2(%2,%arg,%arg2,%1 : i32,i32,i32,i32),
            #fir.interval, %2, %3, ^bb3(%2,%arg2 : i32,i32),
            #fir.upper, %arg, ^bb4(%1 : i32),
            unit, ^bb5]
    ```
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$selector,
      "llvm::ArrayRef<mlir::Attribute>":$compareAttrs,
      "llvm::ArrayRef<mlir::ValueRange>":$cmpOperands,
      "llvm::ArrayRef<mlir::Block *>":$destinations,
      CArg<"llvm::ArrayRef<mlir::ValueRange>", "{}">:$destOperands,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "mlir::Value":$selector,
      "llvm::ArrayRef<mlir::Attribute>":$compareAttrs,
      "llvm::ArrayRef<mlir::Value>":$cmpOpList,
      "llvm::ArrayRef<mlir::Block *>":$destinations,
      CArg<"llvm::ArrayRef<mlir::ValueRange>", "{}">:$destOperands,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>];

  let parser = "return parseSelectCase(parser, result);";

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector().getType() << " [";
    auto cases = (*this)->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
    auto count = getNumConditions();
    for (decltype(count) i = 0; i != count; ++i) {
      if (i)
        p << ", ";
      p << cases[i] << ", ";
      if (!cases[i].isa<mlir::UnitAttr>()) {
        auto caseArgs = *getCompareOperands(i);
        p.printOperand(*caseArgs.begin());
        p << ", ";
        if (cases[i].isa<fir::ClosedIntervalAttr>()) {
          p.printOperand(*(++caseArgs.begin()));
          p << ", ";
        }
      }
      printSuccessorAtIndex(p, i);
    }
    p << ']';
    p.printOptionalAttrDict((*this)->getAttrs(), {getCasesAttr(), getCompareOffsetAttr(),
        getTargetOffsetAttr(), getOperandSegmentSizeAttr()});
  }];

  let verifier = [{
    if (!(getSelector().getType().isa<mlir::IntegerType>() ||
          getSelector().getType().isa<mlir::IndexType>() ||
          getSelector().getType().isa<fir::IntegerType>() ||
          getSelector().getType().isa<fir::LogicalType>() ||
          getSelector().getType().isa<fir::CharacterType>()))
      return emitOpError("must be an integer, character, or logical");
    auto cases = (*this)->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
    auto count = getNumDest();
    if (count == 0)
      return emitOpError("must have at least one successor");
    if (getNumConditions() != count)
      return emitOpError("number of conditions and successors don't match");
    if (compareOffsetSize() != count)
      return emitOpError("incorrect number of compare operand groups");
    if (targetOffsetSize() != count)
      return emitOpError("incorrect number of successor operand groups");
    for (decltype(count) i = 0; i != count; ++i) {
      auto &attr = cases[i];
      if (!(attr.isa<fir::PointIntervalAttr>() ||
            attr.isa<fir::LowerBoundAttr>() ||
            attr.isa<fir::UpperBoundAttr>() ||
            attr.isa<fir::ClosedIntervalAttr>() ||
            attr.isa<mlir::UnitAttr>()))
        return emitOpError("incorrect select case attribute type");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = extraSwitchClassDeclaration#[{
    unsigned compareOffsetSize();
  }];
}

def fir_SelectTypeOp : fir_SwitchTerminatorOp<"select_type"> {
  let summary = "Fortran's SELECT TYPE statement";

  let description = [{
    Similar to `select`, `select_type` provides a way to express Fortran's
    SELECT TYPE construct.  In this case, the type of the selector value
    is matched against a list of type descriptors.  The structure is the
    same as `select`, but `select_type` determines the type of the selector
    variable at runtime to determine the best match.

    ```mlir
      fir.select_type %arg : !fir.box<()> [
          #fir.instance<!fir.type<type1>>, ^bb1(%0 : i32),
          #fir.instance<!fir.type<type2>>, ^bb2(%2 : i32),
          #fir.subsumed<!fir.type<type3>>, ^bb3(%2 : i32),
          #fir.instance<!fir.type<type4>>, ^bb4(%1,%3 : i32,f32),
          unit, ^bb5]
    ```
  }];

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::Value":$selector,
    "llvm::ArrayRef<mlir::Attribute>":$typeOperands,
    "llvm::ArrayRef<mlir::Block *>":$destinations,
    CArg<"llvm::ArrayRef<mlir::ValueRange>", "{}">:$destOperands,
    CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes),
    [{
      $_state.addOperands(selector);
      $_state.addAttribute(getCasesAttr(),
        $_builder.getArrayAttr(typeOperands));
      const auto count = destinations.size();
      for (auto d : destinations)
        $_state.addSuccessors(d);
      const auto opCount = destOperands.size();
      llvm::SmallVector<int32_t, 8> argOffs;
      int32_t sumArgs = 0;
      for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
        if (i < opCount) {
          $_state.addOperands(destOperands[i]);
          const auto argSz = destOperands[i].size();
          argOffs.push_back(argSz);
          sumArgs += argSz;
        } else {
          argOffs.push_back(0);
        }
      }
      $_state.addAttribute(getOperandSegmentSizeAttr(),
        $_builder.getI32VectorAttr({1, 0, sumArgs}));
      $_state.addAttribute(getTargetOffsetAttr(),
        $_builder.getI32VectorAttr(argOffs));
      $_state.addAttributes(attributes);
    }]
  >];

  let parser = "return parseSelectType(parser, result);";

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector().getType() << " [";
    auto cases = (*this)->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
    auto count = getNumConditions();
    for (decltype(count) i = 0; i != count; ++i) {
      if (i)
        p << ", ";
      p << cases[i] << ", ";
      printSuccessorAtIndex(p, i);
    }
    p << ']';
    p.printOptionalAttrDict((*this)->getAttrs(), {getCasesAttr(), getCompareOffsetAttr(),
        getTargetOffsetAttr(), getOperandSegmentSizeAttr()});
  }];

  let verifier = [{
    if (!(getSelector().getType().isa<fir::BoxType>()))
      return emitOpError("must be a boxed type");
    auto cases = (*this)->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
    auto count = getNumDest();
    if (count == 0)
      return emitOpError("must have at least one successor");
    if (getNumConditions() != count)
      return emitOpError("number of conditions and successors don't match");
    if (targetOffsetSize() != count)
      return emitOpError("incorrect number of successor operand groups");
    for (decltype(count) i = 0; i != count; ++i) {
      auto &attr = cases[i];
      if (!(attr.isa<fir::ExactTypeAttr>() || attr.isa<fir::SubclassAttr>() ||
            attr.isa<mlir::UnitAttr>()))
        return emitOpError("invalid type-case alternative");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = extraSwitchClassDeclaration;
}

def fir_UnreachableOp : fir_Op<"unreachable", [Terminator]> {
  let summary = "the unreachable instruction";

  let description = [{
    Terminates a basic block with the assertion that the end of the block
    will never be reached at runtime.  This instruction can be used
    immediately after a call to the Fortran runtime to terminate the
    program, for example.  This instruction corresponds to the LLVM IR
    instruction `unreachable`.

    ```mlir
      fir.unreachable
    ```
  }];

  let parser = "return mlir::success();";

  let printer = "p << getOperationName();";
}

def fir_FirEndOp : fir_Op<"end", [Terminator]> {
  let summary = "the end instruction";

  let description = [{
    The end terminator is a special terminator used inside various FIR
    operations that have regions.  End is thus the custom invisible terminator
    for these operations.  It is implicit and need not appear in the textual
    representation.
  }];
}

def fir_HasValueOp : fir_Op<"has_value", [Terminator, HasParent<"GlobalOp">]> {
  let summary = "terminator for GlobalOp";
  let description = [{
    The terminator for a GlobalOp with a body.

    ```mlir
      global @variable : tuple<i32, f32> {
        %0 = constant 45 : i32
        %1 = constant 100.0 : f32
        %2 = fir.undefined tuple<i32, f32>
        %3 = constant 0 : index
        %4 = fir.insert_value %2, %0, %3 : (tuple<i32, f32>, i32, index) -> tuple<i32, f32>
        %5 = constant 1 : index
        %6 = fir.insert_value %4, %1, %5 : (tuple<i32, f32>, f32, index) -> tuple<i32, f32>
        fir.has_value %6 : tuple<i32, f32>
      }
    ```
  }];

  let arguments = (ins AnyType:$resval);

  let assemblyFormat = "$resval attr-dict `:` type($resval)";
}

//===------------------------------------------------------------------------===//
// Operations on !fir.box<T> type objects
//===------------------------------------------------------------------------===//

def fir_EmboxOp : fir_Op<"embox", [NoSideEffect, AttrSizedOperandSegments]> {
  let summary = "boxes a given reference and (optional) dimension information";

  let description = [{
    Create a boxed reference value. In Fortran, the implementation can require
    extra information about an entity, such as its type, rank, etc.  This
    auxilliary information is packaged and abstracted as a value with box type
    by the calling routine. (In Fortran, these are called descriptors.)

    ```mlir
      %c1 = constant 1 : index
      %c10 = constant 10 : index
      %5 = ... : !fir.ref<!fir.array<10 x i32>>
      %6 = fir.embox %5 : (!fir.ref<!fir.array<10 x i32>>) -> !fir.box<!fir.array<10 x i32>>
    ```

    The descriptor tuple may contain additional implementation-specific
    information through the use of additional attributes.
    Specifically,
        - shape: emboxing an array may require shape information (an array's
          lower bounds and extents may not be known until runtime),
        - slice: an array section can be described with a slice triple,
        - lenParams: for emboxing a derived type with LEN type parameters,
        - accessMap: unused/experimental.
  }];

  let arguments = (ins
    AnyReferenceLike:$memref,
    Optional<AnyShapeType>:$shape,
    Optional<fir_SliceType>:$slice,
    Variadic<AnyIntegerType>:$lenParams,
    OptionalAttr<AffineMapAttr>:$accessMap
  );

  let results = (outs fir_BoxType);

  let builders = [
    OpBuilder<(ins "llvm::ArrayRef<mlir::Type>":$resultTypes,
      "mlir::Value":$memref, CArg<"mlir::Value", "{}">:$shape,
      CArg<"mlir::Value", "{}">:$slice,
      CArg<"mlir::ValueRange", "{}">:$lenParams),
    [{ return build($_builder, $_state, resultTypes, memref, shape, slice,
                    lenParams, mlir::AffineMapAttr{}); }]>
  ];

  let assemblyFormat = [{
    $memref (`(` $shape^ `)`)? (`[` $slice^ `]`)? (`typeparams` $lenParams^)?
      (`map` $accessMap^)? attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    mlir::Value getShape() { return shape(); }
    mlir::Value getSlice() { return slice(); }
    bool hasLenParams() { return !lenParams().empty(); }
    unsigned numLenParams() { return lenParams().size(); }
  }];
}

def fir_ReboxOp : fir_Op<"rebox", [NoSideEffect, AttrSizedOperandSegments]> {
  let summary = "create a box given another box and (optional) dimension information";

  let description = [{
    Create a new boxed reference value from another box. This is meant to be used
    when the taking a reference to part of a boxed value, or to an entire boxed value with
    new shape or type information.

    The new extra information can be:
      - new shape information (new lower bounds, new rank, or new extents.
        New rank/extents can only be provided if the original fir.box is
        contiguous in all dimension but maybe the first one). The shape
        operand must be provided to set new shape information.
      - new type (only for derived types). It is possible to set the dynamic type
        of the new box to one of the parent types of the input box dynamic type.
        Type parameters cannot be changed. This change is reflected in the requested
        result type of the new box.

    A slice argument can be provided to build a reference to part of a boxed value.
    In this case, the shape operand must be absent or be a fir.shift that can be
    used to provide a non default origin for the slice.

    The following example illustrates creating a fir.box for x(10:33:2)
    where x is described by a fir.box and has non default lower bounds,
    and then applying a new 2-dimension shape to this fir.box.

    ```mlir
      %0 = fir.slice %c10, %c33, %c2 : (index, index, index) -> !fir.slice<1>
      %1 = fir.shift %c0 : (index) -> !fir.shift<1>
      %2 = fir.rebox %x(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
      %3 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
      %4 = fir.rebox %2(%3) : (!fir.box<!fir.array<?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
    ```

  }];

  let arguments = (ins
    fir_BoxType:$box,
    Optional<AnyShapeOrShiftType>:$shape,
    Optional<fir_SliceType>:$slice
  );

  let results = (outs fir_BoxType);

  let assemblyFormat = [{
    $box (`(` $shape^ `)`)? (`[` $slice^ `]`)? attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{ return ::verify(*this); }];
}

def fir_EmboxCharOp : fir_Op<"emboxchar", [NoSideEffect]> {
  let summary = "boxes a given CHARACTER reference and its LEN parameter";

  let description = [{
    Create a boxed CHARACTER value. The CHARACTER type has the LEN type
    parameter, the value of which may only be known at runtime.  Therefore,
    a variable of type CHARACTER has both its data reference as well as a
    LEN type parameter.

    ```fortran
      CHARACTER(LEN=10) :: var
    ```
    ```mlir
      %4 = ...         : !fir.ref<!fir.array<10 x !fir.char<1>>>
      %5 = constant 10 : i32
      %6 = fir.emboxchar %4, %5 : (!fir.ref<!fir.array<10 x !fir.char<1>>>, i32) -> !fir.boxchar<1>
    ```

    In the above `%4` is a memory reference to a buffer of 10 CHARACTER units.
    This buffer and its LEN value (10) are wrapped into a pair in `%6`.
  }];

  let arguments = (ins AnyReferenceLike:$memref, AnyIntegerLike:$len);

  let results = (outs fir_BoxCharType);

  let assemblyFormat = [{
    $memref `,` $len attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto eleTy = elementTypeOf(memref().getType());
    if (!eleTy.dyn_cast<CharacterType>())
      return mlir::failure();
    return mlir::success();
  }];
}

def fir_EmboxProcOp : fir_Op<"emboxproc", [NoSideEffect]> {
  let summary = "boxes a given procedure and optional host context";

  let description = [{
    Creates an abstract encapsulation of a PROCEDURE POINTER along with an
    optional pointer to a host instance context. If the pointer is not to an
    internal procedure or the internal procedure does not need a host context
    then the form takes only the procedure's symbol.

    ```mlir
      %0 = fir.emboxproc @f : ((i32) -> i32) -> !fir.boxproc<(i32) -> i32>
    ```

    An internal procedure requiring a host instance for correct execution uses
    the second form. The closure of the host procedure's state is passed as a
    reference to a tuple. It is the responsibility of the host to manage the
    context's values accordingly, up to and including inhibiting register
    promotion of local values.

    ```mlir
      %4 = ... : !fir.ref<tuple<i32, i32>>
      %5 = fir.emboxproc @g, %4 : ((i32) -> i32, !fir.ref<tuple<i32, i32>>) -> !fir.boxproc<(i32) -> i32>
    ```
  }];

  let arguments = (ins SymbolRefAttr:$funcname, AnyReferenceLike:$host);

  let results = (outs fir_BoxProcType);

  let parser = [{
    mlir::SymbolRefAttr procRef;
    if (parser.parseAttribute(procRef, "funcname", result.attributes))
      return mlir::failure();
    bool hasTuple = false;
    mlir::OpAsmParser::OperandType tupleRef;
    if (!parser.parseOptionalComma()) {
      if (parser.parseOperand(tupleRef))
        return mlir::failure();
      hasTuple = true;
    }
    mlir::FunctionType type;
    if (parser.parseColon() ||
        parser.parseLParen() ||
        parser.parseType(type))
      return mlir::failure();
    result.addAttribute("functype", mlir::TypeAttr::get(type));
    if (hasTuple) {
      mlir::Type tupleType;
      if (parser.parseComma() ||
          parser.parseType(tupleType) ||
          parser.resolveOperand(tupleRef, tupleType, result.operands))
        return mlir::failure();
    }
    mlir::Type boxType;
    if (parser.parseRParen() ||
        parser.parseArrow() ||
        parser.parseType(boxType) ||
        parser.addTypesToList(boxType, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << (*this)->getAttr("funcname");
    auto h = host();
    if (h) {
      p << ", ";
      p.printOperand(h);
    }
    p << " : (" << (*this)->getAttr("functype");
    if (h)
      p << ", " << h.getType();
    p << ") -> " << getType();
  }];

  let verifier = [{
    // host bindings (optional) must be a reference to a tuple
    if (auto h = host()) {
      if (auto r = h.getType().dyn_cast<ReferenceType>()) {
        if (!r.getEleTy().dyn_cast<mlir::TupleType>())
          return mlir::failure();
      } else {
        return mlir::failure();
      }
    }
    return mlir::success();
  }];
}

def fir_UnboxOp : fir_SimpleOp<"unbox", [NoSideEffect]> {
  let summary = "unbox the boxed value into a tuple value";

  let description = [{
    Unbox a boxed value into a result of multiple values from the box's
    component data.  The values are, minimally, a reference to the data of the
    entity, the byte-size of one element, the rank, the type descriptor, a set
    of flags (packed in an integer, and an array of dimension information (of
    size rank).

    ```mlir
      %40   = ... : !fir.box<!fir.type<T>>
      %41:6 = fir.unbox %40 : (!fir.box<!fir.type<T>>) -> (!fir.ref<!fir.type<T>>, i32, i32, !fir.tdesc<!fir.type<T>>, i32, !fir.array<? x index>)
    ```
  }];

  let arguments = (ins fir_BoxType:$box);

  let results = (outs
    fir_ReferenceType,  // pointer to data
    AnyIntegerLike,     // size of a data element
    AnyIntegerLike,     // rank of data
    fir_TypeDescType,   // abstract type descriptor
    AnyIntegerLike,     // attribute flags (bitfields)
    fir_SequenceType    // dimension information (if any)
  );
}

def fir_UnboxCharOp : fir_SimpleOp<"unboxchar", [NoSideEffect]> {
  let summary = "unbox a boxchar value into a pair value";

  let description = [{
    Unboxes a value of `boxchar` type into a pair consisting of a memory
    reference to the CHARACTER data and the LEN type parameter.

    ```mlir
      %45   = ... : !fir.boxchar<1>
      %46:2 = fir.unboxchar %45 : (!fir.boxchar<1>) -> (!fir.ref<!fir.character<1>>, i32)
    ```
  }];

  let arguments = (ins fir_BoxCharType:$boxchar);

  let results = (outs fir_ReferenceType, AnyIntegerLike);
}

def fir_UnboxProcOp : fir_SimpleOp<"unboxproc", [NoSideEffect]> {
  let summary = "unbox a boxproc value into a pair value";

  let description = [{
    Unboxes a value of `boxproc` type into a pair consisting of a procedure
    pointer and a pointer to a host context.

    ```mlir
      %47   = ... : !fir.boxproc<() -> i32>
      %48:2 = fir.unboxproc %47 : (!fir.ref<() -> i32>, !fir.ref<tuple<f32, i32>>)
    ```
  }];

  let verifier = [{
    if (auto eleTy = fir::dyn_cast_ptrEleTy(refTuple().getType()))
      if (eleTy.isa<mlir::TupleType>())
        return mlir::success();
    return emitOpError("second output argument has bad type");
  }];

  let arguments = (ins fir_BoxProcType:$boxproc);

  let results = (outs FunctionType, fir_ReferenceType:$refTuple);
}

def fir_BoxAddrOp : fir_SimpleOneResultOp<"box_addr", [NoSideEffect]> {
  let summary = "return a memory reference to the boxed value";

  let description = [{
    This operator is overloaded to work with values of type `box`,
    `boxchar`, and `boxproc`.  The result for each of these
    cases, respectively, is the address of the data, the address of the
    `CHARACTER` data, and the address of the procedure.

    ```mlir
      %51 = fir.box_addr %box : (!fir.box<f64>) -> !fir.ref<f64>
      %52 = fir.box_addr %boxchar : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1>>
      %53 = fir.box_addr %boxproc : (!fir.boxproc<!P>) -> !fir.ref<!P>
    ```
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs AnyReferenceLike);

  let hasFolder = 1;
}

def fir_BoxCharLenOp : fir_SimpleOp<"boxchar_len", [NoSideEffect]> {
  let summary = "return the LEN type parameter from a boxchar value";

  let description = [{
    Extracts the LEN type parameter from a `boxchar` value.

    ```mlir
      %45 = ... : !boxchar<1>  // CHARACTER(20)
      %59 = fir.boxchar_len %45 : (!fir.boxchar<1>) -> i64  // len=20
    ```
  }];

  let arguments = (ins fir_BoxCharType:$val);

  let results = (outs AnyIntegerLike);

  let hasFolder = 1;
}

def fir_BoxDimsOp : fir_Op<"box_dims", [NoSideEffect]> {
  let summary = "return the dynamic dimension information for the boxed value";

  let description = [{
    Returns the triple of lower bound, extent, and stride for `dim` dimension
    of `val`, which must have a `box` type.  The dimensions are enumerated from
    left to right from 0 to rank-1. This operation has undefined behavior if
    `dim` is out of bounds.

    ```mlir
      %c1   = constant 0 : i32
      %52:3 = fir.box_dims %40, %c1 : (!fir.box<!fir.array<*:f64>>, i32) -> (index, index, index)
    ```

    The above is a request to return the left most row (at index 0) triple from
    the box. The triple will be the lower bound, extent, and byte-stride, which
    are the values encoded in a standard descriptor.
  }];

  let arguments = (ins fir_BoxType:$val, AnyIntegerLike:$dim);

  let results = (outs Index, Index, Index);

  let assemblyFormat = [{
    $val `,` $dim attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    mlir::Type getTupleType();
  }];
}

def fir_BoxEleSizeOp : fir_SimpleOneResultOp<"box_elesize", [NoSideEffect]> {
  let summary = "return the size of an element of the boxed value";

  let description = [{
    Returns the size of an element in an entity of `box` type.  This size may
    not be known until runtime.

    ```mlir
      %53 = fir.box_elesize %40 : (!fir.box<f32>, i32) -> i32  // size=4
      %54 = fir.box_elesize %40 : (!fir.box<!fir.array<*:f32>>, i32) -> i32
    ```

    In the above example, `%53` may box an array of REAL values while `%54`
    must box an array of REAL values (with dynamic rank and extent).
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs AnyIntegerLike);
}

def fir_BoxIsAllocOp : fir_SimpleOp<"box_isalloc", [NoSideEffect]> {
  let summary = "is the boxed value an ALLOCATABLE?";

  let description = [{
    Determine if the boxed value was from an ALLOCATABLE entity. This will
    return true if the originating box value was from a `fir.embox` op
    with a mem-ref value that had the type !fir.heap<T>.

    ```mlir
      %r = ... : !fir.heap<i64>
      %b = fir.embox %r : (!fir.heap<i64>) -> !fir.box<i64>
      %a = fir.box_isalloc %b : (!fir.box<i64>) -> i1  // true
    ```

    The canonical descriptor implementation will carry a flag to record if the
    variable is an `ALLOCATABLE`.
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs BoolLike);
}

def fir_BoxIsArrayOp : fir_SimpleOp<"box_isarray", [NoSideEffect]> {
  let summary = "is the boxed value an array?";

  let description = [{
    Determine if the boxed value has a positive (> 0) rank. This will return
    true if the originating box value was from a fir.embox with a memory
    reference value that had the type !fir.array<T> and/or a shape argument.

    ```mlir
      %r = ... : !fir.ref<i64>
      %c_100 = constant 100 : index
      %d = fir.shape %c_100 : (index) -> !fir.shape<1>
      %b = fir.embox %r(%d) : (!fir.ref<i64>, !fir.shape<1>) -> !fir.box<i64>
      %a = fir.box_isarray %b : (!fir.box<i64>) -> i1  // true
    ```
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs BoolLike);
}

def fir_BoxIsPtrOp : fir_SimpleOp<"box_isptr", [NoSideEffect]> {
  let summary = "is the boxed value a POINTER?";

  let description = [{
    Determine if the boxed value was from a POINTER entity.

    ```mlir
      %p = ... : !fir.ptr<i64>
      %b = fir.embox %p : (!fir.ptr<i64>) -> !fir.box<i64>
      %a = fir.box_isptr %b : (!fir.box<i64>) -> i1  // true
    ```
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs BoolLike);
}

def fir_BoxProcHostOp : fir_SimpleOp<"boxproc_host", [NoSideEffect]> {
  let summary = "returns the host instance pointer (or null)";

  let description = [{
    Extract the host context pointer from a boxproc value.

    ```mlir
      %8 = ... : !fir.boxproc<(!fir.ref<!fir.type<T>>) -> i32>
      %9 = fir.boxproc_host %8 : (!fir.boxproc<(!fir.ref<!fir.type<T>>) -> i32>) -> !fir.ref<tuple<i32, i32>>
    ```

    In the example, the reference to the closure over the host procedure's
    variables is returned. This allows an internal procedure to access the
    host's variables. It is up to lowering to determine the contract between
    the host and the internal procedure.
  }];

  let arguments = (ins fir_BoxProcType:$val);

  let results = (outs fir_ReferenceType);
}

def fir_BoxRankOp : fir_SimpleOneResultOp<"box_rank", [NoSideEffect]> {
  let summary = "return the number of dimensions for the boxed value";

  let description = [{
    Return the rank of a value of `box` type.  If the value is scalar, the
    rank is 0.

    ```mlir
      %57 = fir.box_rank %40 : (!fir.box<!fir.array<*:f64>>) -> i32
      %58 = fir.box_rank %41 : (!fir.box<f64>) -> i32
    ```

    The example `%57` shows how one would determine the rank of an array that
    has deferred rank at runtime. This rank should be at least 1. In %58, the
    descriptor may be either an array or a scalar, so the value is nonnegative.
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs AnyIntegerType);
}

def fir_BoxTypeDescOp : fir_SimpleOneResultOp<"box_tdesc", [NoSideEffect]> {
  let summary = "return the type descriptor for the boxed value";

  let description = [{
    Return the opaque type descriptor of a value of `box` type. A type
    descriptor is an implementation defined value that fully describes a type
    to the Fortran runtime.

    ```mlir
      %7 = fir.box_tdesc %41 : (!fir.box<f64>) -> !fir.tdesc<f64>
    ```
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs fir_TypeDescType);
}

//===----------------------------------------------------------------------===//
// Array value operations
//===----------------------------------------------------------------------===//

def fir_ArrayLoadOp : fir_Op<"array_load", [AttrSizedOperandSegments]> {

  let summary = "Load an array as a value.";

  let description = [{
    Load an entire array as a single SSA value.

    ```fortran
      real :: a(o:n,p:m)
      ...
      ... = ... a ...
    ```

    One can use `fir.array_load` to produce an ssa-value that captures an
    immutable value of the entire array `a`, as in the Fortran array expression
    shown above. Subsequent changes to the memory containing the array do not
    alter its composite value. This operation let's one load an array as a
    value while applying a runtime shape, shift, or slice to the memory
    reference, and its semantics guarantee immutability.

    ```mlir
      %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shape<2>
      // load the entire array 'a'
      %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
      // a fir.store here into array %a does not change %v
    ```
  }];

  let arguments = (ins
    Arg<AnyRefOrBox, "", [MemRead]>:$memref,
    Optional<AnyShapeOrShiftType>:$shape,
    Optional<fir_SliceType>:$slice,
    Variadic<AnyIntegerType>:$lenParams
  );

  let results = (outs fir_SequenceType);

  let assemblyFormat = [{
    $memref (`(`$shape^`)`)? (`[`$slice^`]`)? (`typeparams` $lenParams^)? attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    std::vector<mlir::Value> getExtents();
  }];
}

def fir_ArrayFetchOp : fir_Op<"array_fetch", [NoSideEffect]> {

  let summary = "Fetch the value of an element of an array value";

  let description = [{
    Fetch the value of an element in an array value.

    ```fortran
      real :: a(n,m)
      ...
      ... a ...
      ... a(r,s+1) ...
    ```

    One can use `fir.array_fetch` to fetch the (implied) value of `a(i,j)` in
    an array expression as shown above. It can also be used to extract the
    element `a(r,s+1)` in the second expression.

    ```mlir
      %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
      // load the entire array 'a'
      %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
      // fetch the value of one of the array value's elements
      %1 = fir.array_fetch %v, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
    ```

    It is only possible to use `array_fetch` on an `array_load` result value.
  }];

  let arguments = (ins
    fir_SequenceType:$sequence,
    Variadic<AnyCoordinateType>:$indices
  );

  let results = (outs AnyType:$element);

  let assemblyFormat = [{
    $sequence `,` $indices attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto arrTy = sequence().getType().cast<fir::SequenceType>();
    if (indices().size() != arrTy.getDimension())
      return emitOpError("number of indices != dimension of array");
    if (element().getType() != arrTy.getEleTy())
      return emitOpError("return type does not match array");
    if (!isa<fir::ArrayLoadOp>(sequence().getDefiningOp()))
      return emitOpError("argument #0 must be result of fir.array_load");
    return mlir::success();
  }];
}

def fir_ArrayUpdateOp : fir_Op<"array_update", [NoSideEffect]> {

  let summary = "Update the value of an element of an array value";

  let description = [{
    Updates the value of an element in an array value. A new array value is
    returned where all element values of the input array are identical except
    for the selected element which is the value passed in the update.

    ```fortran
      real :: a(n,m)
      ...
      a = ...
    ```

    One can use `fir.array_update` to update the (implied) value of `a(i,j)`
    in an array expression as shown above.

    ```mlir
      %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
      // load the entire array 'a'
      %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
      // update the value of one of the array value's elements
      // %r_{ij} = %f  if (i,j) = (%i,%j),   %v_{ij} otherwise
      %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
      fir.array_merge_store %v, %r to %a : !fir.ref<!fir.array<?x?xf32>>
    ```

    An array value update behaves as if a mapping function from the indices
    to the new value has been added, replacing the previous mapping. These
    mappings can be added to the ssa-value, but will not be materialized in
    memory until the `fir.array_merge_store` is performed.
  }];

  let arguments = (ins
    fir_SequenceType:$sequence,
    AnyType:$merge,
    Variadic<AnyCoordinateType>:$indices
  );

  let results = (outs fir_SequenceType);

  let assemblyFormat = [{
    $sequence `,` $merge `,` $indices attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto arrTy = sequence().getType().cast<fir::SequenceType>();
    if (merge().getType() != arrTy.getEleTy())
      return emitOpError("merged value does not have element type");
    if (indices().size() != arrTy.getDimension())
      return emitOpError("number of indices != dimension of array");
    return mlir::success();
  }];
}

def fir_ArrayMergeStoreOp : fir_Op<"array_merge_store", [
    TypesMatchWith<"type of 'original' matches element type of 'memref'",
                     "memref", "original",
                     "fir::dyn_cast_ptrOrBoxEleTy($_self)">,
    TypesMatchWith<"type of 'sequence' matches element type of 'memref'",
                     "memref", "sequence",
                     "fir::dyn_cast_ptrOrBoxEleTy($_self)">]> {

  let summary = "Store merged array value to memory.";

  let description = [{
    Store a merged array value to memory.

    ```fortran
      real :: a(n,m)
      ...
      a = ...
    ```

    One can use `fir.array_merge_store` to merge/copy the value of `a` in an
    array expression as shown above.

    ```mlir
      %v = fir.array_load %a(%shape) : ...
      %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
      fir.array_merge_store %v, %r to %a : !fir.ref<!fir.array<?x?xf32>>
    ```

    This operation merges the original loaded array value, `%v`, with the
    chained updates, `%r`, and stores the result to the array at address, `%a`.
  }];

  let arguments = (ins
    fir_SequenceType:$original,
    fir_SequenceType:$sequence,
    Arg<AnyRefOrBox, "", [MemWrite]>:$memref
  );

  let assemblyFormat = "$original `,` $sequence `to` $memref attr-dict `:` type($memref)";

  let verifier = [{
    if (!isa<ArrayLoadOp>(original().getDefiningOp()))
       return emitOpError("operand #0 must be result of a fir.array_load op");
    return mlir::success();
  }];
}

//===----------------------------------------------------------------------===//
// Record and array type operations
//===----------------------------------------------------------------------===//

def fir_ArrayCoorOp : fir_Op<"array_coor",
    [NoSideEffect, AttrSizedOperandSegments]> {

  let summary = "Find the coordinate of an element of an array";

  let description = [{
    Compute the location of an element in an array when the shape of the
    array is only known at runtime.

    This operation is intended to capture all the runtime values needed to
    compute the address of an array reference in a single high-level op. Given
    the following Fortran input:

    ```fortran
      real :: a(n,m)
      ...
      ... a(i,j) ...
    ```

    One can use `fir.array_coor` to determine the address of `a(i,j)`.

    ```mlir
      %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
      %1 = fir.array_coor %a(%s) %i, %j : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
    ```
  }];

  let arguments = (ins
    AnyRefOrBox:$memref,
    Optional<AnyShapeOrShiftType>:$shape,
    Optional<fir_SliceType>:$slice,
    Variadic<AnyCoordinateType>:$indices,
    Variadic<AnyIntegerType>:$lenParams
  );

  let results = (outs fir_ReferenceType);

  let assemblyFormat = [{
    $memref (`(`$shape^`)`)? (`[`$slice^`]`)? $indices (`typeparams` $lenParams^)? attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{ return ::verify(*this); }];
}

def fir_CoordinateOp : fir_Op<"coordinate_of", [NoSideEffect]> {

  let summary = "Finds the coordinate (location) of a value in memory";

  let description = [{
    Compute the internal coordinate address starting from a boxed value or
    unboxed memory reference. Returns a memory reference. When computing the
    coordinate of an array element, the rank of the array must be known and
    the number of indexing expressions must equal the rank of the array.

    This operation will apply the access map from a boxed value implicitly.

    Unlike LLVM's GEP instruction, one cannot stride over the outermost
    reference; therefore, the leading 0 index must be omitted.

    ```mlir
      %i = ... : index
      %h = ... : !fir.heap<!fir.array<100 x f32>>
      %p = fir.coordinate_of %h, %i : (!fir.heap<!fir.array<100 x f32>>, index) -> !fir.ref<f32>
    ```

    In the example, `%p` will be a pointer to the `%i`-th f32 value in the
    array `%h`.
  }];

  let arguments = (ins
    AnyRefOrBox:$ref,
    Variadic<AnyCoordinateType>:$coor,
    TypeAttr:$baseType
  );

  let results = (outs fir_ReferenceType);

  let parser =  [{ return parseCoordinateCustom(parser, result); }];
  let printer = [{ ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType,
      "mlir::Value":$ref, "mlir::ValueRange":$coor),
    [{ return build($_builder, $_state, resultType, ref, coor,
           mlir::TypeAttr::get(ref.getType())); }]>,
  ];

  let extraClassDeclaration = [{
    /// Get the type of the base object.
    mlir::Type getBaseType() { return baseType(); }
  }];
}

def fir_ExtractValueOp : fir_OneResultOp<"extract_value", [NoSideEffect]> {
  let summary = "Extract a value from an aggregate SSA-value";

  let description = [{
    Extract a value from an entity with a type composed of tuples, arrays,
    and/or derived types. Returns the value from entity with the type of the
    specified component. Cannot be used on values of `!fir.box` type.
    It can also be used to access complex parts and elements of a character
    string.

    Note that the entity ssa-value must be of compile-time known size in order
    to use this operation.

    ```mlir
      %f = fir.field_index field, !fir.type<X{field:i32}>
      %s = ... : !fir.type<X>
      %v = fir.extract_value %s, %f : (!fir.type<X>, !fir.field) -> i32
    ```
  }];

  let arguments = (ins
    AnyCompositeLike:$adt,
    Variadic<AnyComponentType>:$coor
  );

  let assemblyFormat = [{
    $adt `,` $coor attr-dict `:` functional-type(operands, results)
  }];
}

def fir_FieldIndexOp : fir_OneResultOp<"field_index", [NoSideEffect]> {
  let summary = "create a field index value from a field identifier";

  let description = [{
    Generate a field (offset) value from an identifier.  Field values may be
    lowered into exact offsets when the layout of a Fortran derived type is
    known at compile-time. The type of a field value is `!fir.field` and
    these values can be used with the `fir.coordinate_of`, `fir.extract_value`,
    or `fir.insert_value` instructions to compute (abstract) addresses of
    subobjects.

    ```mlir
      %f = fir.field_index field, !fir.type<X{field:i32}>
    ```
  }];

  let arguments = (ins
    StrAttr:$field_id,
    TypeAttr:$on_type,
    Variadic<AnyIntegerType>:$lenparams
  );

  let parser = [{
    llvm::StringRef fieldName;
    auto &builder = parser.getBuilder();
    mlir::Type recty;
    if (parser.parseOptionalKeyword(&fieldName) ||
        parser.parseComma() ||
        parser.parseType(recty))
      return mlir::failure();
    result.addAttribute(fieldAttrName(), builder.getStringAttr(fieldName));
    if (!recty.dyn_cast<RecordType>())
      return mlir::failure();
    result.addAttribute(typeAttrName(), mlir::TypeAttr::get(recty));
    if (!parser.parseOptionalLParen()) {
      llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> operands;
      llvm::SmallVector<mlir::Type, 8> types;
      auto loc = parser.getNameLoc();
      if (parser.parseOperandList(operands,
                                  mlir::OpAsmParser::Delimiter::None) ||
          parser.parseColonTypeList(types) ||
          parser.parseRParen() ||
          parser.resolveOperands(operands, types, loc, result.operands))
        return mlir::failure();
    }
    mlir::Type fieldType = fir::FieldType::get(builder.getContext());
    if (parser.addTypeToList(fieldType, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' '
      << (*this)->getAttrOfType<mlir::StringAttr>(fieldAttrName()).getValue()
      << ", " << (*this)->getAttr(typeAttrName());
    if (getNumOperands()) {
      p << '(';
      p.printOperands(lenparams());
      auto sep = ") : ";
      for (auto op : lenparams()) {
        p << sep;
        if (op)
          p.printType(op.getType());
        else
          p << "()";
        sep = ", ";
      }
    }
  }];

  let builders = [OpBuilder<(ins "llvm::StringRef":$fieldName,
      "mlir::Type":$recTy, CArg<"mlir::ValueRange","{}">:$operands),
    [{
      $_state.addAttribute(fieldAttrName(),
        $_builder.getStringAttr(fieldName));
      $_state.addAttribute(typeAttrName(), TypeAttr::get(recTy));
      $_state.addOperands(operands);
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef fieldAttrName() { return "field_id"; }
    static constexpr llvm::StringRef typeAttrName() { return "on_type"; }
    llvm::StringRef getFieldName() { return field_id(); }
  }];
}

def fir_ShapeOp : fir_Op<"shape", [NoSideEffect]> {

  let summary = "generate an abstract shape vector of type `!fir.shape`";

  let description = [{
    The arguments are an ordered list of integral type values that define the
    runtime extent of each dimension of an array. The shape information is
    given in the same row-to-column order as Fortran. This abstract shape value
    must be applied to a reified object, so all shape information must be
    specified.  The extent must be nonnegative.

    ```mlir
      %d = fir.shape %row_sz, %col_sz : (index, index) -> !fir.shape<2>
    ```
  }];

  let arguments = (ins Variadic<AnyIntegerType>:$extents);

  let results = (outs fir_ShapeType);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto size = extents().size();
    auto shapeTy = getType().dyn_cast<fir::ShapeType>();
    assert(shapeTy && "must be a shape type");
    if (shapeTy.getRank() != size)
      return emitOpError("shape type rank mismatch");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    std::vector<mlir::Value> getExtents() {
      return {extents().begin(), extents().end()};
    }
  }];
}

def fir_ShapeShiftOp : fir_Op<"shape_shift", [NoSideEffect]> {

  let summary = [{
    generate an abstract shape and shift vector of type `!fir.shapeshift`
  }];

  let description = [{
    The arguments are an ordered list of integral type values that is a multiple
    of 2 in length. Each such pair is defined as: the lower bound and the
    extent for that dimension. The shifted shape information is given in the
    same row-to-column order as Fortran. This abstract shifted shape value must
    be applied to a reified object, so all shifted shape information must be
    specified.  The extent must be nonnegative.

    ```mlir
      %d = fir.shape_shift %lo, %extent : (index, index) -> !fir.shapeshift<1>
    ```
  }];

  let arguments = (ins Variadic<AnyIntegerType>:$pairs);

  let results = (outs fir_ShapeShiftType);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto size = pairs().size();
    if (size < 2 || size > 16 * 2)
      return emitOpError("incorrect number of args");
    if (size % 2 != 0)
      return emitOpError("requires a multiple of 2 args");
    auto shapeTy = getType().dyn_cast<fir::ShapeShiftType>();
    assert(shapeTy && "must be a shape shift type");
    if (shapeTy.getRank() * 2 != size)
      return emitOpError("shape type rank mismatch");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    // Logically unzip the origins from the extent values.
    std::vector<mlir::Value> getOrigins() {
      std::vector<mlir::Value> result;
      for (auto i : llvm::enumerate(pairs()))
        if (!(i.index() & 1))
          result.push_back(i.value());
      return result;
    }

    // Logically unzip the extents from the origin values.
    std::vector<mlir::Value> getExtents() {
      std::vector<mlir::Value> result;
      for (auto i : llvm::enumerate(pairs()))
        if (i.index() & 1)
          result.push_back(i.value());
      return result;
    }
  }];
}

def fir_ShiftOp : fir_Op<"shift", [NoSideEffect]> {

  let summary = "generate an abstract shift vector of type `!fir.shift`";

  let description = [{
    The arguments are an ordered list of integral type values that define the
    runtime lower bound of each dimension of an array. The shape information is
    given in the same row-to-column order as Fortran. This abstract shift value
    must be applied to a reified object, so all shift information must be
    specified.

    ```mlir
      %d = fir.shift %row_lb, %col_lb : (index, index) -> !fir.shift<2>
    ```
  }];

  let arguments = (ins Variadic<AnyIntegerType>:$origins);

  let results = (outs fir_ShiftType);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto size = origins().size();
    auto shiftTy = getType().dyn_cast<fir::ShiftType>();
    assert(shiftTy && "must be a shift type");
    if (shiftTy.getRank() != size)
      return emitOpError("shift type rank mismatch");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    std::vector<mlir::Value> getOrigins() {
      return {origins().begin(), origins().end()};
    }
  }];
}

def fir_SliceOp : fir_Op<"slice", [NoSideEffect, AttrSizedOperandSegments]> {

  let summary = "generate an abstract slice vector of type `!fir.slice`";

  let description = [{
    The array slicing arguments are an ordered list of integral type values
    that must be a multiple of 3 in length.  Each such triple is defined as:
    the lower bound, the upper bound, and the stride for that dimension, as in
    Fortran syntax. Both bounds are inclusive. The array slice information is
    given in the same row-to-column order as Fortran. This abstract slice value
    must be applied to a reified object, so all slice information must be
    specified.  The extent must be nonnegative and the stride must not be zero.

    ```mlir
      %d = fir.slice %lo, %hi, %step : (index, index, index) -> !fir.slice<1>
    ```

    To support generalized slicing of Fortran's dynamic derived types, a slice
    op can be given a component path (narrowing from the product type of the
    original array to the specific elemental type of the sliced projection).

    ```mlir
      %fld = fir.field_index component, !fir.type<t{...component:ct...}>
      %d = fir.slice %lo, %hi, %step path %fld : (index, index, index, !fir.field) -> !fir.slice<1>
    ```
  }];

  let arguments = (ins
    Variadic<AnyCoordinateType>:$triples,
    Variadic<AnyComponentType>:$fields
  );

  let results = (outs fir_SliceType);

  let assemblyFormat = [{
    $triples (`path` $fields^)? attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto size = triples().size();
    if (size < 3 || size > 16 * 3)
      return emitOpError("incorrect number of args for triple");
    if (size % 3 != 0)
      return emitOpError("requires a multiple of 3 args");
    auto sliceTy = getType().dyn_cast<fir::SliceType>();
    assert(sliceTy && "must be a slice type");
    if (sliceTy.getRank() * 3 != size)
      return emitOpError("slice type rank mismatch");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    unsigned getOutRank() { return getOutputRank(triples()); }
    static unsigned getOutputRank(mlir::ValueRange triples);
  }];
}

def fir_InsertValueOp : fir_OneResultOp<"insert_value", [NoSideEffect]> {
  let summary = "insert a new sub-value into a copy of an existing aggregate";

  let description = [{
    Insert a value into an entity with a type composed of tuples, arrays,
    and/or derived types. Returns a new ssa value with the same type as the
    original entity. Cannot be used on values of `!fir.box` type.
    It can also be used to set complex parts and elements of a character
    string.

    Note that the entity ssa-value must be of compile-time known size in order
    to use this operation.

    ```mlir
      %a = ... : !fir.array<10xtuple<i32, f32>>
      %f = ... : f32
      %o = ... : i32
      %c = constant 1 : i32
      %b = fir.insert_value %a, %f, %o, %c : (!fir.array<10x20xtuple<i32, f32>>, f32, i32, i32) -> !fir.array<10x20xtuple<i32, f32>>
    ```
  }];

  let arguments = (ins AnyCompositeLike:$adt, AnyType:$val,
                       Variadic<AnyComponentType>:$coor);
  let results = (outs AnyCompositeLike);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

def fir_InsertOnRangeOp : fir_OneResultOp<"insert_on_range", [NoSideEffect]> {
  let summary = "insert sub-value into a range on an existing sequence";

  let description = [{
    Insert a constant value into an entity with an array type. Returns a
    new ssa value where the range of offsets from the original array have been
    replaced with the constant. The result is an array type entity.
  }];

  let arguments = (ins fir_SequenceType:$seq, AnyType:$val,
                       Variadic<Index>:$coor);
  let results = (outs fir_SequenceType);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def fir_LenParamIndexOp : fir_OneResultOp<"len_param_index", [NoSideEffect]> {
  let summary =
    "create a field index value from a LEN type parameter identifier";

  let description = [{
    Generate a LEN parameter (offset) value from an LEN parameter identifier.
    The type of a LEN parameter value is `!fir.len` and these values can be
    used with the `fir.coordinate_of` instructions to compute (abstract)
    addresses of LEN parameters.

    ```mlir
      %e = fir.len_param_index len1, !fir.type<X(len1:i32)>
      %p = ... : !fir.box<!fir.type<X>>
      %q = fir.coordinate_of %p, %e : (!fir.box<!fir.type<X>>, !fir.len) -> !fir.ref<i32>
    ```
  }];

  let arguments = (ins StrAttr:$field_id, TypeAttr:$on_type);

  let parser = [{
    llvm::StringRef fieldName;
    auto &builder = parser.getBuilder();
    mlir::Type recty;
    if (parser.parseOptionalKeyword(&fieldName) ||
        parser.parseComma() ||
        parser.parseType(recty))
      return mlir::failure();
    result.addAttribute(fieldAttrName(), builder.getStringAttr(fieldName));
    if (!recty.dyn_cast<RecordType>())
      return mlir::failure();
    result.addAttribute(typeAttrName(), mlir::TypeAttr::get(recty));
    mlir::Type lenType = fir::LenType::get(builder.getContext());
    if (parser.addTypeToList(lenType, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' '
      << (*this)->getAttrOfType<mlir::StringAttr>(fieldAttrName()).getValue()
      << ", " << (*this)->getAttr(typeAttrName());
  }];

  let builders = [OpBuilder<(ins "llvm::StringRef":$fieldName,
      "mlir::Type":$recTy),
    [{
      $_state.addAttribute(fieldAttrName(), $_builder.getStringAttr(fieldName));
      $_state.addAttribute(typeAttrName(), TypeAttr::get(recTy));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef fieldAttrName() { return "field_id"; }
    static constexpr llvm::StringRef typeAttrName() { return "on_type"; }
    mlir::Type getOnType() {
      return (*this)->getAttrOfType<TypeAttr>(typeAttrName()).getValue();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Fortran loops
//===----------------------------------------------------------------------===//

def fir_ResultOp : fir_Op<"result",
    [NoSideEffect, ReturnLike, Terminator,
     ParentOneOf<["IfOp", "DoLoopOp", "IterWhileOp"]>]> {
  let summary = "special terminator for use in fir region operations";

  let description = [{
    Result takes a list of ssa-values produced in the block and forwards them
    as a result to the operation that owns the region of the block. The
    operation can retain the values or return them to its parent block
    depending upon its semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* do nothing */ }]>];

  let assemblyFormat = "($results^ `:` type($results))? attr-dict";

  let verifier = [{ return ::verify(*this); }];
}

def FirRegionTerminator : SingleBlockImplicitTerminator<"ResultOp">;

class region_Op<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic,
    !listconcat(traits, [FirRegionTerminator, RecursiveSideEffects])> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def fir_DoLoopOp : region_Op<"do_loop",
    [DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "generalized loop operation";
  let description = [{
    Generalized high-level looping construct. This operation is similar to
    MLIR's `scf.for`.

    ```mlir
      %l = constant 0 : index
      %u = constant 9 : index
      %s = constant 1 : index
      fir.do_loop %i = %l to %u step %s unordered {
        %x = fir.convert %i : (index) -> i32
        %v = fir.call @compute(%x) : (i32) -> f32
        %p = fir.coordinate_of %A, %i : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
        fir.store %v to %p : !fir.ref<f32>
      }
    ```

    The above example iterates over the interval `[%l, %u]`. The unordered
    keyword indicates that the iterations can be executed in any order.
  }];

  let arguments = (ins
    Index:$lowerBound,
    Index:$upperBound,
    Index:$step,
    Variadic<AnyType>:$initArgs,
    OptionalAttr<UnitAttr>:$unordered,
    OptionalAttr<UnitAttr>:$finalValue
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$lowerBound, "mlir::Value":$upperBound,
      "mlir::Value":$step, CArg<"bool", "false">:$unordered,
      CArg<"bool", "false">:$finalCountValue,
      CArg<"mlir::ValueRange", "llvm::None">:$iterArgs,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = [{
    mlir::Value getInductionVar() { return getBody()->getArgument(0); }
    mlir::OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    mlir::Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }
    mlir::Operation::operand_range getIterOperands() {
      return getOperands().drop_front(getNumControlOperands());
    }

    void setLowerBound(Value bound) { (*this)->setOperand(0, bound); }
    void setUpperBound(Value bound) { (*this)->setOperand(1, bound); }
    void setStep(Value step) { (*this)->setOperand(2, step); }

    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - 1;
    }
    /// Number of operands controlling the loop: lb, ub, step
    unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return (*this)->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return (*this)->getNumOperands() - getNumControlOperands();
    }

    /// Get the body of the loop
    mlir::Block *getBody() { return &region().front(); }

    void setUnordered() {
      unorderedAttr(mlir::UnitAttr::get(getContext()));
    }

    mlir::BlockArgument iterArgToBlockArg(mlir::Value iterArg);
    void resultToSourceOps(llvm::SmallVectorImpl<mlir::Value> &results,
                           unsigned resultNum);
    mlir::Value blockArgToSourceOp(unsigned blockArgNum);
  }];
}

def fir_IfOp : region_Op<"if", [NoRegionArguments]> {
  let summary = "if-then-else conditional operation";
  let description = [{
    Used to conditionally execute operations. This operation is the FIR
    dialect's version of `loop.if`.

    ```mlir
      %56 = ... : i1
      %78 = ... : !fir.ref<!T>
      fir.if %56 {
        fir.store %76 to %78 : !fir.ref<!T>
      } else {
        fir.store %77 to %78 : !fir.ref<!T>
      }
    ```
  }];

  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region
    SizedRegion<1>:$thenRegion,
    AnyRegion:$elseRegion
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$cond, "bool":$withElseRegion)>,
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::Value":$cond,
        "bool":$withElseRegion)>
  ];

  let extraClassDeclaration = [{
    mlir::OpBuilder getThenBodyBuilder() {
      assert(!thenRegion().empty() && "Unexpected empty 'where' region.");
      mlir::Block &body = thenRegion().front();
      return mlir::OpBuilder(&body, std::prev(body.end()));
    }
    mlir::OpBuilder getElseBodyBuilder() {
      assert(!elseRegion().empty() && "Unexpected empty 'other' region.");
      mlir::Block &body = elseRegion().front();
      return mlir::OpBuilder(&body, std::prev(body.end()));
    }

    void resultToSourceOps(llvm::SmallVectorImpl<mlir::Value> &results,
                           unsigned resultNum);
  }];
}

def fir_IterWhileOp : region_Op<"iterate_while",
    [DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "DO loop with early exit condition";
  let description = [{
    This single-entry, single-exit looping construct is useful for lowering
    counted loops that can exit early such as, for instance, implied-DO loops.
    It is very similar to `fir::DoLoopOp` with the addition that it requires
    a single loop-carried bool value that signals an early exit condition to
    the operation. A `true` disposition means the next loop iteration should
    proceed. A `false` indicates that the `fir.iterate_while` operation should
    terminate and return its iteration arguments. This is a degenerate counted
    loop in that the loop is not guaranteed to execute all iterations.

    An example iterate_while that returns the counter value, the early
    termination condition, and an extra loop-carried value is shown here. This
    loop counts from %lo to %up (inclusive), stepping by %c1, so long as the
    early exit (%ok) is true. The iter_args %sh value is also carried by the
    loop. The result triple is the values of %i=phi(%lo,%i+%c1),
    %ok=phi(%okIn,%okNew), and %sh=phi(%shIn,%shNew) from the last executed
    iteration.

    ```mlir
      %v:3 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok = %okIn) iter_args(%sh = %shIn) -> (index, i1, i16) {
        %shNew = fir.call @bar(%sh) : (i16) -> i16
        %okNew = fir.call @foo(%sh) : (i16) -> i1
        fir.result %i, %okNew, %shNew : index, i1, i16
      }
    ```
  }];

  let arguments = (ins
    Index:$lowerBound,
    Index:$upperBound,
    Index:$step,
    I1:$iterateIn,
    Variadic<AnyType>:$initArgs,
    OptionalAttr<UnitAttr>:$finalValue
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$lowerBound, "mlir::Value":$upperBound,
      "mlir::Value":$step, "mlir::Value":$iterate,
      CArg<"bool", "false">:$finalCountValue,
      CArg<"mlir::ValueRange", "llvm::None">:$iterArgs,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = [{
    mlir::Block *getBody() { return &region().front(); }
    mlir::Value getIterateVar() { return getBody()->getArgument(1); }
    mlir::Value getInductionVar() { return getBody()->getArgument(0); }
    mlir::OpBuilder getBodyBuilder() {
      return mlir::OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    mlir::Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }
    mlir::Operation::operand_range getIterOperands() {
      return getOperands().drop_front(getNumControlOperands());
    }

    void setLowerBound(Value bound) { (*this)->setOperand(0, bound); }
    void setUpperBound(Value bound) { (*this)->setOperand(1, bound); }
    void setStep(mlir::Value step) { (*this)->setOperand(2, step); }

    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - 1;
    }
    /// Number of operands controlling the loop
    unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return (*this)->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return (*this)->getNumOperands() - getNumControlOperands();
    }

    mlir::BlockArgument iterArgToBlockArg(mlir::Value iterArg);
    void resultToSourceOps(llvm::SmallVectorImpl<mlir::Value> &results,
                           unsigned resultNum);
    mlir::Value blockArgToSourceOp(unsigned blockArgNum);
  }];
}

//===----------------------------------------------------------------------===//
// Procedure call operations
//===----------------------------------------------------------------------===//

def fir_CallOp : fir_Op<"call", [CallOpInterface]> {
  let summary = "call a procedure";

  let description = [{
    Call the specified function or function reference.

    Provides a custom parser and pretty printer to allow a more readable syntax
    in the FIR dialect, e.g. `fir.call @sub(%12)` or `fir.call %20(%22,%23)`.

    ```mlir
      %a = fir.call %funcref(%arg0) : (!fir.ref<f32>) -> f32
      %b = fir.call @function(%arg1, %arg2) : (!fir.ref<f32>, !fir.ref<f32>) -> f32
    ```
  }];

  let arguments = (ins
    OptionalAttr<SymbolRefAttr>:$callee,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>);

  let parser = "return parseCallOp(parser, result);";
  let printer = "printCallOp(p, *this);";

  let builders = [
    OpBuilder<(ins "mlir::FuncOp":$callee,
        CArg<"mlir::ValueRange", "{}">:$operands),
    [{
      $_state.addOperands(operands);
      $_state.addAttribute(calleeAttrName($_state.name),
        $_builder.getSymbolRefAttr(callee));
      $_state.addTypes(callee.getType().getResults());
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee,
        "llvm::ArrayRef<mlir::Type>":$results,
        CArg<"mlir::ValueRange", "{}">:$operands),
    [{
      $_state.addOperands(operands);
      $_state.addAttribute(calleeAttrName($_state.name), callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "llvm::StringRef":$callee,
        "llvm::ArrayRef<mlir::Type>":$results,
        CArg<"mlir::ValueRange", "{}">:$operands),
    [{
      build($_builder, $_state, $_builder.getSymbolRefAttr(callee), results,
            operands);
    }]>];

  let extraClassDeclaration = [{
    mlir::FunctionType getFunctionType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      if (calleeAttr())
        return {arg_operand_begin(), arg_operand_end()};
      return {arg_operand_begin() + 1, arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      if (auto calling = calleeAttr())
        return calling;
      return getOperand(0);
    }
  }];
}

def fir_DispatchOp : fir_Op<"dispatch", []> {
  let summary = "call a type-bound procedure";

  let description = [{
    Perform a dynamic dispatch on the method name via the dispatch table
    associated with the first argument.  The attribute 'pass_arg_pos' can be
    used to select a dispatch argument other than the first one.

    ```mlir
      %r = fir.dispatch methodA(%o) : (!fir.box<none>) -> i32
    ```
  }];

  let arguments = (ins
    StrAttr:$method,
    fir_BoxType:$object,
    Variadic<AnyType>:$args
  );

  let results = (outs Variadic<AnyType>);

  let parser = [{
    mlir::FunctionType calleeType;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
    auto calleeLoc = parser.getNameLoc();
    llvm::StringRef calleeName;
    if (failed(parser.parseOptionalKeyword(&calleeName))) {
      mlir::StringAttr calleeAttr;
      if (parser.parseAttribute(calleeAttr, "method", result.attributes))
        return mlir::failure();
    } else {
      result.addAttribute(methodAttrName(result.name),
          parser.getBuilder().getStringAttr(calleeName));
    }
    if (parser.parseOperandList(operands,
                                mlir::OpAsmParser::Delimiter::Paren) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(calleeType) ||
        parser.addTypesToList(calleeType.getResults(), result.types) ||
        parser.resolveOperands(
            operands, calleeType.getInputs(), calleeLoc, result.operands))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << methodAttr() << '(';
    p.printOperand(object());
    if (!args().empty()) {
      p << ", ";
      p.printOperands(args());
    }
    p << ") : ";
    p.printFunctionalType((*this)->getOperandTypes(),
        (*this)->getResultTypes());
  }];

  let extraClassDeclaration = [{
    mlir::FunctionType getFunctionType();
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    // operand[0] is the object (of box type)
    operand_iterator arg_operand_begin() { return operand_begin() + 1; }
    operand_iterator arg_operand_end() { return operand_end(); }
    static constexpr llvm::StringRef passArgAttrName() {
      return "pass_arg_pos";
    }
    unsigned passArgPos();
  }];
}

// Constant operations that support Fortran

def fir_StringLitOp : fir_Op<"string_lit", [NoSideEffect]> {
  let summary = "create a string literal constant";

  let description = [{
    An FIR constant that represents a sequence of characters that correspond
    to Fortran's CHARACTER type, including a LEN.  We support CHARACTER values
    of different KINDs (different constant sizes).

    ```mlir
      %1 = fir.string_lit "Hello, World!"(13) : !fir.char<1> // ASCII
      %2 = fir.string_lit [158, 2345](2) : !fir.char<2>      // Wide chars
    ```
  }];

  let results = (outs fir_CharacterType);

  let parser = [{
    auto &builder = parser.getBuilder();
    mlir::Attribute val;
    mlir::NamedAttrList attrs;
    llvm::SMLoc trailingTypeLoc;
    if (parser.parseAttribute(val, "fake", attrs))
      return mlir::failure();
    if (auto v = val.dyn_cast<mlir::StringAttr>())
      result.attributes.push_back(builder.getNamedAttr(value(), v));
    else if (auto v = val.dyn_cast<mlir::ArrayAttr>())
      result.attributes.push_back(builder.getNamedAttr(xlist(), v));
    else
      return parser.emitError(parser.getCurrentLocation(),
                              "found an invalid constant");
    mlir::IntegerAttr sz;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(sz, size(), result.attributes) ||
        parser.parseRParen() ||
        parser.getCurrentLocation(&trailingTypeLoc) ||
        parser.parseColonType(type))
      return mlir::failure();
    auto charTy = type.dyn_cast<fir::CharacterType>();
    if (!charTy)
      return parser.emitError(trailingTypeLoc,
                              "must have character type");
    type = fir::CharacterType::get(builder.getContext(), charTy.getFKind(),
                                   sz.getInt());
    if (!type || parser.addTypesToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getValue() << '(';
    p << getSize().cast<mlir::IntegerAttr>().getValue() << ") : ";
    p.printType(getType());
  }];

  let verifier = [{
    if (getSize().cast<mlir::IntegerAttr>().getValue().isNegative())
      return emitOpError("size must be non-negative");
    if (auto xl = (*this)->getAttr(xlist())) {
      auto xList = xl.cast<mlir::ArrayAttr>();
      for (auto a : xList)
        if (!a.isa<mlir::IntegerAttr>())
	    return emitOpError("values in list must be integers");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr const char *size() { return "size"; }
    static constexpr const char *value() { return "value"; }
    static constexpr const char *xlist() { return "xlist"; }

    // Get the LEN attribute of this character constant
    mlir::Attribute getSize() { return (*this)->getAttr(size()); }
    // Get the string value of this character constant
    mlir::Attribute getValue() {
      if (auto attr = (*this)->getAttr(value()))
        return attr;
      return (*this)->getAttr(xlist());
    }

    /// Is this a wide character literal (1 character > 8 bits)
    bool isWideValue();
  }];
}

// Complex operations

class fir_ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic,
           !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])>,
    Results<(outs AnyType)> {
  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{ return printBinaryOp(this->getOperation(), p); }];
}

class fir_UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_Op<mnemonic,
             !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])>,
      Results<(outs AnyType)> {
  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{ return printUnaryOp(this->getOperation(), p); }];
}

class RealUnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_UnaryArithmeticOp<mnemonic, traits>,
      Arguments<(ins AnyRealLike:$operand)>;

def fir_NegfOp : RealUnaryArithmeticOp<"negf">;

class RealArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_ArithmeticOp<mnemonic, traits>,
      Arguments<(ins AnyRealLike:$lhs, AnyRealLike:$rhs)>;

def fir_ModfOp : RealArithmeticOp<"modf">;

def fir_CmpfOp : fir_Op<"cmpf",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape]> {
  let summary = "floating-point comparison operator";

  let description = [{
    Extends the standard floating-point comparison to handle the extended
    floating-point types found in FIR.
  }];

  let arguments = (ins AnyRealLike:$lhs, AnyRealLike:$rhs);

  let results = (outs AnyLogicalLike);

  let builders = [OpBuilder<(ins "mlir::CmpFPredicate":$predicate,
    "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      buildCmpFOp($_builder, $_state, predicate, lhs, rhs);
  }]>];

  let parser = [{ return parseCmpfOp(parser, result); }];

  let printer = [{ printCmpfOp(p, *this); }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getPredicateAttrName() {
      return "predicate";
    }
    static CmpFPredicate getPredicateByName(llvm::StringRef name);

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)(*this)->getAttrOfType<mlir::IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];
}

def fir_ConstcOp : fir_Op<"constc", [NoSideEffect]> {
  let summary = "create a complex constant";

  let description = [{
    A complex constant. Similar to the standard dialect complex type, but this
    extension allows constants with APFloat values that are not supported in
    the standard dialect.
  }];

  let results = (outs fir_ComplexType);

  let parser = [{
    fir::RealAttr realp;
    fir::RealAttr imagp;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(realp, realAttrName(), result.attributes) ||
        parser.parseComma() ||
        parser.parseAttribute(imagp, imagAttrName(), result.attributes) ||
        parser.parseRParen() ||
        parser.parseColonType(type) ||
        parser.addTypesToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << " (0x";
    auto f1 = (*this)->getAttr(realAttrName()).cast<mlir::FloatAttr>();
    auto i1 = f1.getValue().bitcastToAPInt();
    p.getStream().write_hex(i1.getZExtValue());
    p << ", 0x";
    auto f2 = (*this)->getAttr(imagAttrName()).cast<mlir::FloatAttr>();
    auto i2 = f2.getValue().bitcastToAPInt();
    p.getStream().write_hex(i2.getZExtValue());
    p << ") : ";
    p.printType(getType());
  }];

  let verifier = [{
    if (!getType().isa<fir::ComplexType>())
      return emitOpError("must be a !fir.complex type");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef realAttrName() { return "real"; }
    static constexpr llvm::StringRef imagAttrName() { return "imaginary"; }

    mlir::Attribute getReal() { return (*this)->getAttr(realAttrName()); }
    mlir::Attribute getImaginary() { return (*this)->getAttr(imagAttrName()); }
  }];
}

class ComplexUnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_UnaryArithmeticOp<mnemonic, traits>,
      Arguments<(ins fir_ComplexType:$operand)>;

def fir_NegcOp : ComplexUnaryArithmeticOp<"negc">;

class ComplexArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_ArithmeticOp<mnemonic, traits>,
      Arguments<(ins fir_ComplexType:$lhs, fir_ComplexType:$rhs)>;

def fir_AddcOp : ComplexArithmeticOp<"addc", [Commutative]>;
def fir_SubcOp : ComplexArithmeticOp<"subc">;
def fir_MulcOp : ComplexArithmeticOp<"mulc", [Commutative]>;
def fir_DivcOp : ComplexArithmeticOp<"divc">;
// Pow is a builtin call and not a primitive

def fir_CmpcOp : fir_Op<"cmpc",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape]> {
  let summary = "complex floating-point comparison operator";

  let description = [{
    A complex comparison to handle complex types found in FIR.
  }];

  let arguments = (ins fir_ComplexType:$lhs, fir_ComplexType:$rhs);

  let results = (outs AnyLogicalLike);

  let parser = "return parseCmpcOp(parser, result);";

  let printer = "printCmpcOp(p, *this);";

  let builders = [OpBuilder<(ins "mlir::CmpFPredicate":$predicate,
    "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      buildCmpCOp($_builder, $_state, predicate, lhs, rhs);
  }]>];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getPredicateAttrName() {
      return "predicate";
    }

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)(*this)->getAttrOfType<mlir::IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];
}

// Other misc. operations

def fir_AddrOfOp : fir_OneResultOp<"address_of", [NoSideEffect]> {
  let summary = "convert a symbol to an SSA value";

  let description = [{
    Convert a symbol (a function or global reference) to an SSA-value to be
    used in other Operations. References to Fortran symbols are distinguished
    via this operation from other arbitrary constant values.

    ```mlir
      %p = fir.address_of(@symbol) : !fir.ref<f64>
    ```
  }];

  let arguments = (ins SymbolRefAttr:$symbol);

  let results = (outs AnyAddressableLike:$resTy);

  let assemblyFormat = "`(` $symbol `)` attr-dict `:` type($resTy)";
}

def fir_ConvertOp : fir_OneResultOp<"convert", [NoSideEffect]> {
  let summary = "encapsulates all Fortran scalar type conversions";

  let description = [{
    Generalized type conversion. Convert the ssa value from type T to type U.
    Not all pairs of types have conversions. When types T and U are the same
    type, this instruction is a NOP and may be folded away.

    ```mlir
      %v = ... : i64
      %w = fir.convert %v : (i64) -> i32
    ```

    The example truncates the value `%v` from an i64 to an i32.
  }];

  let arguments = (ins AnyType:$value);

  let assemblyFormat = [{
    $value attr-dict `:` functional-type($value, results)
  }];

  let hasFolder = 1;

  let verifier = [{
    auto inType = value().getType();
    auto outType = getType();
    if (inType == outType)
      return mlir::success();
    if ((isPointerCompatible(inType) && isPointerCompatible(outType)) ||
        (isIntegerCompatible(inType) && isIntegerCompatible(outType)) ||
        (isIntegerCompatible(inType) && isFloatCompatible(outType)) ||
        (isFloatCompatible(inType) && isIntegerCompatible(outType)) ||
        (isFloatCompatible(inType) && isFloatCompatible(outType)) ||
        (isIntegerCompatible(inType) && isPointerCompatible(outType)) ||
        (isPointerCompatible(inType) && isIntegerCompatible(outType)) ||
        (inType.isa<fir::BoxType>() && outType.isa<fir::BoxType>()) ||
        (fir::isa_complex(inType) && fir::isa_complex(outType)))
      return mlir::success();
    return emitOpError("invalid type conversion");
  }];

  let extraClassDeclaration = [{
    static bool isIntegerCompatible(mlir::Type ty);
    static bool isFloatCompatible(mlir::Type ty);
    static bool isPointerCompatible(mlir::Type ty);
  }];
  let hasCanonicalizer = 1;
}

def FortranTypeAttr : Attr<And<[CPred<"$_self.isa<TypeAttr>()">,
    Or<[CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::CharacterType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::ComplexType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::IntegerType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::LogicalType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::RealType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::RecordType>()">]>]>,
    "Fortran surface type"> {
  let storageType = [{ TypeAttr }];
  let returnType = "Type";
  let convertFromStorage = "$_self.getValue().cast<Type>()";
}

def fir_GenTypeDescOp : fir_OneResultOp<"gentypedesc", [NoSideEffect]> {
  let summary = "generate a type descriptor for a given type";
  let description = [{
    Generates a constant object that is an abstract type descriptor of the
    specified type.  The meta-type of a type descriptor for the type `T`
    is `!fir.tdesc<T>`.

    ```mlir
      !T = type !fir.type<T{...}>
      %t = fir.gentypedesc !T  // returns value of !fir.tdesc<!T>
    ```
  }];

  let arguments = (ins FortranTypeAttr:$in_type);

  let parser = [{
    mlir::Type intype;
    if (parser.parseType(intype))
      return mlir::failure();
    result.addAttribute("in_type", mlir::TypeAttr::get(intype));
    mlir::Type restype = TypeDescType::get(intype);
    if (parser.addTypeToList(restype, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << (*this)->getAttr("in_type");
    p.printOptionalAttrDict((*this)->getAttrs(), {"in_type"});
  }];

  let builders = [OpBuilder<(ins "mlir::TypeAttr":$inty)>];

  let verifier = [{
    mlir::Type resultTy = getType();
    if (auto tdesc = resultTy.dyn_cast<TypeDescType>()) {
      if (tdesc.getOfTy() != getInType())
        return emitOpError("wrapped type mismatched");
    } else {
      return emitOpError("must be !fir.tdesc type");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    mlir::Type getInType() {
      // get the type that the type descriptor describes
      return (*this)->getAttrOfType<mlir::TypeAttr>("in_type").getValue();
    }
  }];
}

def fir_NoReassocOp : fir_OneResultOp<"no_reassoc",
    [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "synthetic op to prevent reassociation";
  let description = [{
    Primitive operation meant to intrusively prevent operator reassociation.
    The operation is otherwise a nop and the value returned is the same as the
    argument.

    The presence of this operation prevents any local optimizations. In the
    example below, this would prevent possibly replacing the multiply and add
    operations with a single FMA operation.

    ```mlir
      %98 = mulf %96, %97 : f32
      %99 = fir.no_reassoc %98 : f32
      %a0 = addf %99, %95 : f32
    ```
  }];

  let arguments = (ins AnyType:$val);

  let assemblyFormat = "$val attr-dict `:` type($val)";
}

class AtMostRegion<int numBlocks> : Region<
  CPred<"$_self.getBlocks().size() <= " # numBlocks>,
  "region with " # numBlocks # " blocks">;

def fir_GlobalOp : fir_Op<"global", [IsolatedFromAbove, Symbol]> {
  let summary = "Global data";
  let description = [{
    A global variable or constant with initial values.

    The example creates a global variable (writable) named
    `@_QV_Mquark_Vvarble` with some initial values. The initializer should
    conform to the variable's type.

    ```mlir
      fir.global @_QV_Mquark_Vvarble : tuple<i32, f32> {
        %1 = constant 1 : i32
        %2 = constant 2.0 : f32
        %3 = fir.undefined tuple<i32, f32>
        %z = constant 0 : index
        %o = constant 1 : index
        %4 = fir.insert_value %3, %1, %z : (tuple<i32, f32>, i32, index) -> tuple<i32, f32>
        %5 = fir.insert_value %4, %2, %o : (tuple<i32, f32>, f32, index) -> tuple<i32, f32>
        fir.has_value %5 : tuple<i32, f32>
      }
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    SymbolRefAttr:$symref,
    TypeAttr:$type,
    OptionalAttr<AnyAttr>:$initVal,
    OptionalAttr<UnitAttr>:$constant,
    OptionalAttr<StrAttr>:$linkName
  );

  let regions = (region AtMostRegion<1>:$region);

  let parser = "return parseGlobalOp(parser, result);";

  let printer = [{
    p << getOperationName();
    if (linkName().hasValue())
      p << ' ' << linkName().getValue();
    p << ' ';
    p.printAttributeWithoutType((*this)->getAttr(symbolAttrName()));
    if (auto val = getValueOrNull())
      p << '(' << val << ')';
    if (constantAttr())
      p << " constant";
    p << " : ";
    p.printType(getType());
    if (hasInitializationBody())
      p.printRegion((*this)->getRegion(0), /*printEntryBlockArgs=*/false,
                    /*printBlockTerminators=*/true);
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::Type":$type,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "llvm::StringRef":$name, "bool":$isConstant,
      "mlir::Type":$type,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::Type":$type,
      CArg<"mlir::StringAttr", "{}">:$linkage,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "llvm::StringRef":$name, "bool":$isConstant,
      "mlir::Type":$type, CArg<"mlir::StringAttr", "{}">:$linkage,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::Type":$type,
      "mlir::Attribute":$initVal, CArg<"mlir::StringAttr", "{}">:$linkage,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "llvm::StringRef":$name, "bool":$isConstant,
      "mlir::Type":$type, "mlir::Attribute":$initVal,
      CArg<"mlir::StringAttr", "{}">:$linkage,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef symbolAttrName() { return "symref"; }
    static constexpr llvm::StringRef linkageAttrName() { return "linkName"; }

    /// The printable type of the global
    mlir::Type getType() {
      return typeAttr().getValue();
    }

    /// The semantic type of the global
    mlir::Type resultType() {
      return fir::AllocaOp::wrapResultType(getType());
    }

    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() { return initVal().getValueOr(Attribute()); }

    /// Append the next initializer value to the `GlobalOp` to construct
    /// the variable's initial value.
    void appendInitialValue(mlir::Operation *op);

    /// A GlobalOp has one region.
    mlir::Region &getRegion() { return (*this)->getRegion(0); }

    /// A GlobalOp has one block.
    mlir::Block &getBlock() { return getRegion().front(); }

    /// Determine if `linkage` is a supported keyword
    static mlir::ParseResult verifyValidLinkage(StringRef linkage);

    bool hasInitializationBody() {
      return ((*this)->getNumRegions() == 1) && !getRegion().empty() &&
        !isa<fir::FirEndOp>(getBlock().front());
    }

    mlir::FlatSymbolRefAttr getSymbol() {
      return mlir::FlatSymbolRefAttr::get(getContext(),
          (*this)->getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()).getValue());
    }
  }];
}

def fir_GlobalLenOp : fir_Op<"global_len", []> {
  let summary = "map a LEN parameter to a global";
  let description = [{
    A global entity (that is not an automatic data object) can have extra LEN
    parameter (compile-time) constants associated with the instance's type.
    These values can be bound to the global instance used `fir.global_len`.

    ```mlir
      global @g : !fir.type<t(len1:i32)> {
        fir.global_len len1, 10 : i32
        %1 = fir.undefined : !fir.type<t(len1:i32)>
        return %1 : !fir.type<t(len1:i32)>
      }
    ```
  }];

  let arguments = (ins StrAttr:$lenparam, APIntAttr:$intval);

  let parser = [{
    llvm::StringRef fieldName;
    if (failed(parser.parseOptionalKeyword(&fieldName))) {
      mlir::StringAttr fieldAttr;
      if (parser.parseAttribute(fieldAttr, lenParamAttrName(),
                                result.attributes))
        return mlir::failure();
    } else {
      result.addAttribute(lenParamAttrName(),
          parser.getBuilder().getStringAttr(fieldName));
    }
    mlir::IntegerAttr constant;
    if (parser.parseComma() ||
        parser.parseAttribute(constant, intAttrName(), result.attributes))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << (*this)->getAttr(lenParamAttrName())
      << ", " << (*this)->getAttr(intAttrName());
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef lenParamAttrName() { return "lenparam"; }
    static constexpr llvm::StringRef intAttrName() { return "intval"; }
  }];
}

def ImplicitFirTerminator : SingleBlockImplicitTerminator<"FirEndOp">;

def fir_DispatchTableOp : fir_Op<"dispatch_table",
    [IsolatedFromAbove, Symbol, ImplicitFirTerminator]> {
  let summary = "Dispatch table definition";

  let description = [{
    Define a dispatch table for a derived type with type-bound procedures.

    A dispatch table is an untyped symbol that contains a list of associations
    between method identifiers and corresponding `FuncOp` symbols.

    The ordering of associations in the map is determined by the front-end.

    ```mlir
      fir.dispatch_table @_QDTMquuzTfoo {
        fir.dt_entry method1, @_QFNMquuzTfooPmethod1AfooR
        fir.dt_entry method2, @_QFNMquuzTfooPmethod2AfooII
      }
    ```
  }];

  let parser = [{
    // Parse the name as a symbol reference attribute.
    SymbolRefAttr nameAttr;
    if (parser.parseAttribute(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                              result.attributes))
      return failure();

    // Convert the parsed name attr into a string attr.
    result.attributes.set(mlir::SymbolTable::getSymbolAttrName(),
      parser.getBuilder().getStringAttr(nameAttr.getRootReference()));

    // Parse the optional table body.
    mlir::Region *body = result.addRegion();
    OptionalParseResult parseResult = parser.parseOptionalRegion(*body);
    if (parseResult.hasValue() && failed(*parseResult))
      return mlir::failure();

    ensureTerminator(*body, parser.getBuilder(), result.location);
    return mlir::success();
  }];

  let printer = [{
    auto tableName = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName()).getValue();
    p << getOperationName() << " @" << tableName;

    Region &body = (*this)->getRegion(0);
    if (!body.empty())
      p.printRegion(body, /*printEntryBlockArgs=*/false,
                          /*printBlockTerminators=*/false);
  }];

  let verifier = [{
    for (auto &op : getBlock())
      if (!(isa<fir::DTEntryOp>(op) || isa<fir::FirEndOp>(op)))
        return emitOpError("dispatch table must contain dt_entry");
    return mlir::success();
  }];

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::Type":$type,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs),
    [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                           $_builder.getStringAttr(name));
      $_state.addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Append a dispatch table entry to the table.
    void appendTableEntry(mlir::Operation *op);

    mlir::Region &getRegion() {
      return (*this)->getRegion(0);
    }

    mlir::Block &getBlock() {
      return getRegion().front();
    }
  }];
}

def fir_DTEntryOp : fir_Op<"dt_entry", []> {
  let summary = "map entry in a dispatch table";

  let description = [{
    An entry in a dispatch table.  Allows a function symbol to be bound
    to a specifier method identifier.  A dispatch operation uses the dynamic
    type of a distinguished argument to determine an exact dispatch table
    and uses the method identifier to select the type-bound procedure to
    be called.

    ```mlir
      fir.dt_entry method_name, @uniquedProcedure
    ```
  }];

  let arguments = (ins StrAttr:$method, SymbolRefAttr:$proc);

  let parser = [{
    llvm::StringRef methodName;
    // allow `methodName` or `"methodName"`
    if (failed(parser.parseOptionalKeyword(&methodName))) {
      mlir::StringAttr methodAttr;
      if (parser.parseAttribute(methodAttr, "method",
                                result.attributes))
        return mlir::failure();
    } else {
      result.addAttribute(methodAttrName(result.name),
          parser.getBuilder().getStringAttr(methodName));
    }
    mlir::SymbolRefAttr calleeAttr;
    if (parser.parseComma() ||
        parser.parseAttribute(calleeAttr, "proc", result.attributes))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << methodAttr() << ", "
      << procAttr();
  }];
}

def fir_AbsentOp : fir_OneResultOp<"absent", [NoSideEffect]> {
  let summary = "create value to be passed for absent optional function argument";
  let description = [{
    Given the type of a function argument, create a value that will signal that
    an optional argument is absent in the call. On the caller side, fir.is_present
    can be used to query if the value of an optional argument was created with
    a fir.absent operation.
    It is undefined to use a value that was created by a fir.absent op in any other
    operation than fir.call and fir.is_present.
    ```mlir
      %1 = fir.absent fir.box<fir.array<?xf32>>
      fir.call @_QPfoo(%1) : (fir.box<fir.array<?xf32>>) -> ()
    ```
  }];

  let results = (outs AnyRefOrBoxLike:$intype);

  let assemblyFormat = "type($intype) attr-dict";
}

def fir_IsPresentOp : fir_SimpleOp<"is_present", [NoSideEffect]> {
  let summary = "is this optional function argument present?";

  let description = [{
    Determine if an optional function argument is PRESENT (i.e. that it was not
    created by a fir.absent op on the caller side).
    ```mlir
      func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>) {
        %0 = fir.is_present %arg0 : (!fir.box<!fir.array<?xf32>>) -> i1
        ...
    ```
  }];

  let arguments = (ins AnyRefOrBoxLike:$val);

  let results = (outs BoolLike);
}

#endif
