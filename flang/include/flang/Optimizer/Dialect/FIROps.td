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

#ifndef FIR_DIALECT_FIR_OPS
#define FIR_DIALECT_FIR_OPS

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def fir_Dialect : Dialect {
  let name = "fir";
  let cppNamespace = "::fir";
}

// Types and predicates

def fir_Type : Type<CPred<"fir::isa_fir_or_std_type($_self)">,
    "FIR dialect type">;

// Fortran intrinsic types
def fir_CharacterType : Type<CPred<"$_self.isa<fir::CharacterType>()">,
    "FIR character type">;
def fir_ComplexType : Type<CPred<"$_self.isa<fir::CplxType>()">,
    "FIR complex type">;
def fir_IntegerType : Type<CPred<"$_self.isa<fir::IntType>()">,
    "FIR integer type">;
def fir_LogicalType : Type<CPred<"$_self.isa<fir::LogicalType>()">,
    "FIR logical type">;
def fir_RealType : Type<CPred<"$_self.isa<fir::RealType>()">,
    "FIR real type">;

// Generalized FIR and standard dialect types representing intrinsic types
def AnyIntegerLike : TypeConstraint<Or<[SignlessIntegerLike.predicate,
    fir_IntegerType.predicate]>, "any integer">;
def AnyLogicalLike : TypeConstraint<Or<[BoolLike.predicate,
    fir_LogicalType.predicate]>, "any logical">;
def AnyRealLike : TypeConstraint<Or<[FloatLike.predicate,
    fir_RealType.predicate]>, "any real">;
def AnyIntegerType : Type<AnyIntegerLike.predicate, "any integer">;

// Fortran derived (user defined) type
def fir_RecordType : Type<CPred<"$_self.isa<fir::RecordType>()">,
    "FIR derived type">;

// Fortran array attribute
def fir_SequenceType : Type<CPred<"$_self.isa<fir::SequenceType>()">,
    "array type">;

// Composable types
def AnyCompositeLike : TypeConstraint<Or<[fir_RecordType.predicate,
    fir_SequenceType.predicate, fir_ComplexType.predicate,
    IsTupleTypePred]>, "any composite">;

// Reference to an entity type
def fir_ReferenceType : Type<CPred<"$_self.isa<fir::ReferenceType>()">,
    "reference type">;

// Reference to an ALLOCATABLE attribute type
def fir_HeapType : Type<CPred<"$_self.isa<fir::HeapType>()">,
    "allocatable type">;

// Reference to a POINTER attribute type
def fir_PointerType : Type<CPred<"$_self.isa<fir::PointerType>()">,
    "pointer type">;

// Reference types
def AnyReferenceLike : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate]>, "any reference">;

// A descriptor tuple (captures a reference to an entity and other information)
def fir_BoxType : Type<CPred<"$_self.isa<fir::BoxType>()">, "box type">;

// CHARACTER type descriptor. A pair of a data reference and a LEN value.
def fir_BoxCharType : Type<CPred<"$_self.isa<fir::BoxCharType>()">,
    "box character type">;

// PROCEDURE POINTER descriptor. A pair that can capture a host closure.
def fir_BoxProcType : Type<CPred<"$_self.isa<fir::BoxProcType>()">,
    "box procedure type">;

def AnyBoxLike : TypeConstraint<Or<[fir_BoxType.predicate,
    fir_BoxCharType.predicate, fir_BoxProcType.predicate]>, "any box">;

def AnyRefOrBox : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate, fir_BoxType.predicate]>,
    "any reference or box">;

// A vector of Fortran triple notation describing a multidimensional array
def fir_DimsType : Type<CPred<"$_self.isa<fir::DimsType>()">, "dim type">;
def AnyEmboxLike : TypeConstraint<Or<[AnySignlessInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_DimsType.predicate]>,
    "any legal embox argument type">;
def AnyEmboxArg : Type<AnyEmboxLike.predicate, "embox argument type">;

// A type descriptor's type
def fir_TypeDescType : Type<CPred<"$_self.isa<fir::TypeDescType>()">,
    "type desc type">;

// A field (in a RecordType) argument's type
def fir_FieldType : Type<CPred<"$_self.isa<fir::FieldType>()">, "field type">;

// A LEN parameter (in a RecordType) argument's type
def fir_LenType : Type<CPred<"$_self.isa<fir::LenType>()">,
    "LEN parameter type">;

def AnyComponentLike : TypeConstraint<Or<[AnySignlessInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_FieldType.predicate]>,
    "any coordinate index">;
def AnyComponentType : Type<AnyComponentLike.predicate, "coordinate type">;

def AnyCoordinateLike : TypeConstraint<Or<[AnySignlessInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_FieldType.predicate,
    fir_LenType.predicate]>, "any coordinate index">;
def AnyCoordinateType : Type<AnyCoordinateLike.predicate, "coordinate type">;

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
def fir_AllocateOpBuilder : OpBuilder<
  "OpBuilder &builder, OperationState &result, Type inType,"
  "ValueRange lenParams = {}, ValueRange sizes = {},"
  "ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addTypes(getRefTy(inType));
    result.addAttribute("in_type", TypeAttr::get(inType));
    result.addOperands(sizes);
    result.addAttributes(attributes);
  }]>;

def fir_NamedAllocateOpBuilder : OpBuilder<
  "OpBuilder &builder, OperationState &result, Type inType, StringRef name,"
  "ValueRange lenParams = {}, ValueRange sizes = {},"
  "ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addTypes(getRefTy(inType));
    result.addAttribute("in_type", TypeAttr::get(inType));
    result.addAttribute("name", builder.getStringAttr(name));
    result.addOperands(sizes);
    result.addAttributes(attributes);
  }]>;

def fir_OneResultOpBuilder : OpBuilder<
  "OpBuilder &, OperationState &result, Type resultType,"
  "ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType)
      result.addTypes(resultType);
    result.addOperands(operands);
    result.addAttributes(attributes);
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

class fir_AllocatableOp<string mnemonic, list<OpTrait> traits = []> :
    fir_AllocatableBaseOp<mnemonic,
	!listconcat(traits, [MemoryEffects<[MemAlloc]>])>,
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
    p << getOperationName() << ' ' << getAttr(inType());
    if (hasLenParams()) {
      // print the LEN parameters to a derived type in parens
      p << '(' << getLenParams() << " : " << getLenParams().getTypes() << ')';
    }
    // print the shape of the allocation (if any); all must be index type
    for (auto sh : getShapeOperands()) {
      p << ", ";
      p.printOperand(sh);
    }
    p.printOptionalAttrDict(getAttrs(), {inType(), lenpName()});
  }];

  string extraAllocClassDeclaration = [{
    static constexpr llvm::StringRef inType() { return "in_type"; }
    static constexpr llvm::StringRef lenpName() { return "len_param_count"; }
    mlir::Type getAllocatedType();
    
    bool hasLenParams() { return bool{getAttr(lenpName())}; }
    
    unsigned numLenParams() {
      if (auto val = getAttrOfType<mlir::IntegerAttr>(lenpName()))
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
      return getAttrOfType<mlir::TypeAttr>(inType()).getValue();
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

def fir_AllocaOp : fir_AllocatableOp<"alloca"> {
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
  }];

  let results = (outs fir_ReferenceType);

  let verifier = allocVerify#[{
    mlir::Type outType = getType();
    if (!outType.isa<fir::ReferenceType>())
      return emitOpError("must be a !fir.ref type");
    return mlir::success();
  }];

  let extraClassDeclaration = extraAllocClassDeclaration#[{
    static mlir::Type wrapResultType(mlir::Type intype);
  }];
}

def fir_LoadOp : fir_OneResultOp<"load", [MemoryEffects<[MemRead]>]> {
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

  let arguments = (ins AnyReferenceLike:$memref);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value refVal",
    [{
      if (!refVal) {
        mlir::emitError(result.location, "LoadOp has null argument");
        return;
      }
      auto refTy = refVal.getType().cast<fir::ReferenceType>();
      result.addOperands(refVal);
      result.addTypes(refTy.getEleTy());
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
    p.printOptionalAttrDict(getAttrs(), {});
    p << " : " << memref().getType();
  }];

  let extraClassDeclaration = [{
    static mlir::ParseResult getElementOf(mlir::Type &ele, mlir::Type ref);
  }];
}

def fir_StoreOp : fir_Op<"store", [MemoryEffects<[MemWrite]>]> {
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

  let arguments = (ins AnyType:$value, AnyReferenceLike:$memref);

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
    p.printOptionalAttrDict(getAttrs(), {});
    p << " : " << memref().getType();
  }];

  let verifier = [{
    if (value().getType() != fir::dyn_cast_ptrEleTy(memref().getType()))
      return emitOpError("store value type must match memory reference type");
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

def fir_AllocMemOp : fir_AllocatableOp<"allocmem"> {
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

  let arguments = (ins fir_HeapType:$heapref);

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
      return getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).size();
    }

    // The selector is the value being tested to determine the destination
    mlir::Value getSelector() { return selector(); }
    mlir::Value getSelector(llvm::ArrayRef<mlir::Value> operands) {
      return operands[0];
    }

    // The number of blocks that may be branched to
    unsigned getNumDest() { return getOperation()->getNumSuccessors(); }

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
  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value selector,"
    "ArrayRef<int64_t> compareOperands, ArrayRef<Block *> destinations,"
    "ArrayRef<ValueRange> destOperands = {},"
    "ArrayRef<NamedAttribute> attributes = {}",
    [{
      result.addOperands(selector);
      llvm::SmallVector<mlir::Attribute, 8> ivalues;
      for (auto iv : compareOperands)
        ivalues.push_back(builder.getI64IntegerAttr(iv));
      ivalues.push_back(builder.getUnitAttr());
      result.addAttribute(getCasesAttr(), builder.getArrayAttr(ivalues));
      const auto count = destinations.size();
      for (auto d : destinations)
        result.addSuccessors(d);
      const auto opCount = destOperands.size();
      llvm::SmallVector<int32_t, 8> argOffs;
      int32_t sumArgs = 0;
      for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
        if (i < opCount) {
          result.addOperands(destOperands[i]);
          const auto argSz = destOperands[i].size();
          argOffs.push_back(argSz);
          sumArgs += argSz;
        } else {
          argOffs.push_back(0);
        }
      }
      result.addAttribute(getOperandSegmentSizeAttr(),
                          builder.getI32VectorAttr({1, 0, sumArgs}));
      result.addAttribute(getTargetOffsetAttr(),
                          builder.getI32VectorAttr(argOffs));
      result.addAttributes(attributes);
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
    auto cases = getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
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
    p.printOptionalAttrDict(getAttrs(), {getCasesAttr(), getCompareOffsetAttr(),
        getTargetOffsetAttr(), getOperandSegmentSizeAttr()});
  }];

  let verifier = [{
    if (!(getSelector().getType().isa<mlir::IntegerType>() ||
          getSelector().getType().isa<mlir::IndexType>() ||
          getSelector().getType().isa<fir::IntType>()))
      return emitOpError("must be an integer");
    auto cases = getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
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
    OpBuilder<"OpBuilder &builder, OperationState &result, Value selector,"
    "ArrayRef<mlir::Attribute> compareAttrs, ArrayRef<ValueRange> cmpOperands,"
    "ArrayRef<Block *> destinations, ArrayRef<ValueRange> destOperands = {},"
    "ArrayRef<NamedAttribute> attributes = {}">,
    OpBuilder<"OpBuilder &builder, OperationState &result, Value selector,"
    "ArrayRef<mlir::Attribute> compareAttrs, ArrayRef<Value> cmpOpList,"
    "ArrayRef<Block *> destinations, ArrayRef<ValueRange> destOperands = {},"
    "ArrayRef<NamedAttribute> attributes = {}">];

  let parser = "return parseSelectCase(parser, result);";

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector().getType() << " [";
    auto cases = getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
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
    p.printOptionalAttrDict(getAttrs(), {getCasesAttr(), getCompareOffsetAttr(),
        getTargetOffsetAttr(), getOperandSegmentSizeAttr()});
  }];

  let verifier = [{
    if (!(getSelector().getType().isa<mlir::IntegerType>() ||
          getSelector().getType().isa<mlir::IndexType>() ||
          getSelector().getType().isa<fir::IntType>() ||
          getSelector().getType().isa<fir::LogicalType>() ||
          getSelector().getType().isa<fir::CharacterType>()))
      return emitOpError("must be an integer, character, or logical");
    auto cases = getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
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
  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value selector,"
    "ArrayRef<mlir::Attribute> typeOperands,"
    "ArrayRef<Block *> destinations, ArrayRef<ValueRange> destOperands = {},"
    "ArrayRef<NamedAttribute> attributes = {}",
    [{
      result.addOperands(selector);
      result.addAttribute(getCasesAttr(), builder.getArrayAttr(typeOperands));
      const auto count = destinations.size();
      for (auto d : destinations)
        result.addSuccessors(d);
      const auto opCount = destOperands.size();
      llvm::SmallVector<int32_t, 8> argOffs;
      int32_t sumArgs = 0;
      for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
        if (i < opCount) {
          result.addOperands(destOperands[i]);
          const auto argSz = destOperands[i].size();
          argOffs.push_back(argSz);
          sumArgs += argSz;
        } else {
          argOffs.push_back(0);
        }
      }
      result.addAttribute(getOperandSegmentSizeAttr(),
                          builder.getI32VectorAttr({1, 0, sumArgs}));
      result.addAttribute(getTargetOffsetAttr(),
                          builder.getI32VectorAttr(argOffs));
      result.addAttributes(attributes);
    }]
  >];

  let parser = "return parseSelectType(parser, result);";

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector().getType() << " [";
    auto cases = getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
    auto count = getNumConditions();
    for (decltype(count) i = 0; i != count; ++i) {
      if (i)
        p << ", ";
      p << cases[i] << ", ";
      printSuccessorAtIndex(p, i);
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs(), {getCasesAttr(), getCompareOffsetAttr(),
        getTargetOffsetAttr(), getOperandSegmentSizeAttr()});
  }];

  let verifier = [{
    if (!(getSelector().getType().isa<fir::BoxType>()))
      return emitOpError("must be a boxed type");
    auto cases = getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
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

// Operations on !fir.box<T> type objects

def fir_EmboxOp : fir_Op<"embox", [NoSideEffect]> {
  let summary = "boxes a given reference and (optional) dimension information";

  let description = [{
    Create a boxed reference value. In Fortran, the implementation can require
    extra information about an entity, such as its type, rank, etc.  This
    auxilliary information is packaged and abstracted as a value with box type
    by the calling routine. (In Fortran, these are called descriptors.)

    ```mlir
      %c1 = constant 1 : index
      %c10 = constant 10 : index
      %4 = fir.dims(%c1, %c10, %c1) : (index, index, index) -> !fir.dims<1>
      %5 = ... : !fir.ref<!fir.array<10 x i32>>
      %6 = fir.embox %5, %4 : (!fir.ref<!fir.array<10 x i32>>, !fir.dims<1>) -> !fir.box<!fir.array<10 x i32>>
    ```

    The descriptor tuple may contain additional implementation-specific
    information through the use of additional attributes.
  }];

  let arguments = (ins AnyReferenceLike:$memref, Variadic<AnyEmboxArg>:$args);

  let results = (outs fir_BoxType);

  let parser = "return parseEmboxOp(parser, result);";

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(memref());
    if (hasLenParams()) {
      p << '(';
      p.printOperands(getLenParams());
      p << ')';
    }
    if (getNumOperands() == 2) {
      p << ", ";
      p.printOperands(dims());
    } else if (auto map = getAttr(layoutName())) {
      p << " [" << map << ']';
    }
    p.printOptionalAttrDict(getAttrs(), {layoutName(), lenpName()});
    p << " : ";
    p.printFunctionalType(getOperation());
  }];

  let verifier = [{
    if (hasLenParams()) {
      auto lenParams = numLenParams();
      auto eleTy = fir::dyn_cast_ptrEleTy(memref().getType());
      if (!eleTy)
        return emitOpError("must embox a memory reference type");
      if (auto rt = eleTy.dyn_cast<fir::RecordType>()) {
        if (lenParams != rt.getNumLenParams())
          return emitOpError("number of LEN params does not correspond"
                             " to the !fir.type type");
      } else {
        return emitOpError("LEN parameters require !fir.type type");
      }
      for (auto lp : getLenParams())
        if (lp.getType().isa<fir::DimsType>())
          return emitOpError("LEN parameters must be integral type");
    }
    if (dims().size() == 0) {
      // Ok. If there is no dims and no layout map, then emboxing a scalar.
      // TODO: Should the type be enforced? It already must agree.
    } else if (dims().size() == 1) {
      auto d = *dims().begin();
      if (!d.getType().isa<fir::DimsType>())
        return emitOpError("dimension argument must have !fir.dims type");
    } else {
      return emitOpError("embox can only have one !fir.dim argument");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef layoutName() { return "layout_map"; }
    static constexpr llvm::StringRef lenpName() { return "len_param_count"; }
    bool hasLenParams() { return bool{getAttr(lenpName())}; }
    unsigned numLenParams() {
      if (auto x = getAttrOfType<mlir::IntegerAttr>(lenpName()))
        return x.getInt();
      return 0;
    }
    operand_range getLenParams() {
      return {operand_begin(), operand_begin() + numLenParams()};
    }
    operand_range dims() {
      return {operand_begin() + numLenParams() + 1, operand_end()};
    }
  }];
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
    p << getOperationName() << ' ' << getAttr("funcname");
    auto h = host();
    if (h) {
      p << ", ";
      p.printOperand(h);
    }
    p << " : (" << getAttr("functype");
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
      %41:6 = fir.unbox %40 : (!fir.box<!fir.type<T>>) -> (!fir.ref<!fir.type<T>>, i32, i32, !fir.tdesc<!fir.type<T>>, i32, !fir.dims<4>)
    ```
  }];

  let arguments = (ins fir_BoxType:$box);

  let results = (outs
    fir_ReferenceType,  // pointer to data
    AnyIntegerLike,     // size of a data element
    AnyIntegerLike,     // rank of data
    fir_TypeDescType,   // abstract type descriptor
    AnyIntegerLike,     // attribute flags (bitfields)
    fir_DimsType        // dimension information (if any)
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
      %52:3 = fir.box_dims %40, %c1 : (!fir.box<!fir.array<*:f64>>, i32) -> (i32, i32, i32)
    ```

    The above is a request to return the left most row (at index 0) triple from
    the box. The triple will be the lower bound, upper bound, and stride.
  }];

  let arguments = (ins fir_BoxType:$val, AnyIntegerLike:$dim);

  let results = (outs AnyIntegerLike, AnyIntegerLike, AnyIntegerLike);

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
    reference value that had the type !fir.array<T> and/or a dims argument.

    ```mlir
      %r = ... : !fir.ref<i64>
      %d = fir.gendims(1, 100, 1) : (i32, i32, i32) -> !fir.dims<1>
      %b = fir.embox %r, %d : (!fir.ref<i64>, !fir.dims<1>) -> !fir.box<i64>
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

// Record and array type operations

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

  let arguments = (ins AnyRefOrBox:$ref, Variadic<AnyCoordinateType>:$coor);

  let results = (outs fir_ReferenceType);

  let parser = "return parseCoordinateOp(parser, result);";

  let printer = [{
    p << getOperationName() << ' ' << getOperation()->getOperands();
    p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{baseType()});
    p << " : ";
    p.printFunctionalType(getOperation()->getOperandTypes(),
        getOperation()->getResultTypes());
  }];
  
  let verifier = [{
    auto refTy = ref().getType();
    if (fir::isa_ref_type(refTy)) {
      auto eleTy = fir::dyn_cast_ptrEleTy(refTy);
      if (auto arrTy = eleTy.dyn_cast<fir::SequenceType>()) {
        if (arrTy.hasUnknownShape())
          return emitOpError("cannot find coordinate in unknown shape");
        if (arrTy.getConstantRows() < arrTy.getDimension() - 1)
          return emitOpError("cannot find coordinate with unknown extents");
      }
    }
    // Recovering a LEN type parameter only makes sense from a boxed value
    for (auto co : coor())
      if (dyn_cast_or_null<LenParamIndexOp>(co.getDefiningOp())) {
        if (getNumOperands() != 2)
          return emitOpError("len_param_index must be last argument");
        if (!ref().getType().dyn_cast<BoxType>())
          return emitOpError("len_param_index must be used on box type");
      }
    if (auto attr = getAttr(CoordinateOp::baseType())) {
      if (!attr.isa<mlir::TypeAttr>())
        return emitOpError("improperly constructed");
    } else {
      return emitOpError("must have base type");
    }
    return mlir::success();
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "Type type, Value ref, ValueRange coor,"
	      "ArrayRef<NamedAttribute> attrs = {}">,
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "Type type, ValueRange operands,"
	      "ArrayRef<NamedAttribute> attrs = {}">];
	      
  let extraClassDeclaration = [{
    static constexpr llvm::StringRef baseType() { return "base_type"; }
    mlir::Type getBaseType();
  }];
}

def fir_ExtractValueOp : fir_OneResultOp<"extract_value", [NoSideEffect]> {
  let summary = "Extract a value from an aggregate SSA-value";

  let description = [{
    Extract a value from an entity with a type composed of tuples, arrays,
    and/or derived types. Returns the value from entity with the type of the
    specified component. Cannot be used on values of `!fir.box` type.

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
          parser.parseRParen() ||
          parser.parseColonTypeList(types) ||
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
      << getAttrOfType<mlir::StringAttr>(fieldAttrName()).getValue() << ", "
      << getAttr(typeAttrName());
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, StringRef fieldName,"
    "Type recTy, ValueRange operands = {}",
    [{
      result.addAttribute(fieldAttrName(), builder.getStringAttr(fieldName));
      result.addAttribute(typeAttrName(), TypeAttr::get(recTy));
      result.addOperands(operands);
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef fieldAttrName() { return "field_id"; }
    static constexpr llvm::StringRef typeAttrName() { return "on_type"; }
  }];
}

def fir_GenDimsOp : fir_OneResultOp<"gendims", [NoSideEffect]> {

  let summary = "generate a value of type `!fir.dims`";

  let description = [{
    The arguments are an ordered list of integral type values that is a
    multiple of 3 in length.  Each such triple is defined as: the lower
    index, the extent, and the stride for that dimension. The dimension
    information is given in the same row-to-column order as Fortran. This
    abstract dimension value must describe a reified object, so all dimension
    information must be specified.  The extent must be nonnegative and the
    stride must not be zero.

    ```mlir
      %d = fir.gendims %lo, %ext, %str : (index, index, index) -> !fir.dims<1>
    ```
  }];

  let arguments = (ins Variadic<AnyIntegerType>:$triples);

  let results = (outs fir_DimsType);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto size = triples().size();
    if (size < 1 || size > 16 * 3)
      return emitOpError("incorrect number of args");
    if (size % 3 != 0)
      return emitOpError("requires a multiple of 3 args");
    return mlir::success();
  }];
}

def fir_InsertValueOp : fir_OneResultOp<"insert_value", [NoSideEffect]> {
  let summary = "insert a new sub-value into a copy of an existing aggregate";

  let description = [{
    Insert a value from an entity with a type composed of tuples, arrays,
    and/or derived types. Returns a new ssa value with the same type as the
    original entity. Cannot be used on values of `!fir.box` type.

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
      << getAttrOfType<mlir::StringAttr>(fieldAttrName()).getValue() << ", "
      << getAttr(typeAttrName());
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, StringRef fieldName, Type recTy",
    [{
      result.addAttribute(fieldAttrName(), builder.getStringAttr(fieldName));
      result.addAttribute(typeAttrName(), TypeAttr::get(recTy));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef fieldAttrName() { return "field_id"; }
    static constexpr llvm::StringRef typeAttrName() { return "on_type"; }
    mlir::Type getOnType() {
      return getAttrOfType<TypeAttr>(typeAttrName()).getValue();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Fortran loops
//===----------------------------------------------------------------------===//

def fir_ResultOp : fir_Op<"result",
    [NoSideEffect, ReturnLike, Terminator,
     ParentOneOf<["WhereOp", "LoopOp", "IterWhileOp"]>]> {
  let summary = "special terminator for use in fir region operations";

  let description = [{
    Result takes a list of ssa-values produced in the block and forwards them
    as a result to the operation that owns the region of the block. The
    operation can retain the values or return them to its parent block
    depending upon its semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result", "/* do nothing */">
  ];

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

def fir_LoopOp : region_Op<"do_loop",
    [DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "generalized loop operation";
  let description = [{
    Generalized high-level looping construct. This operation is similar to
    MLIR's `loop.for`.

    ```mlir
      %l = constant 0 : index
      %u = constant 9 : index
      %s = constant 1 : index
      fir.do_loop %i = %l to %u step %s unordered {
        %x = fir.convert %i : (index) -> i32
        %v = fir.call @compute(%x) : (i32) -> f32
        %p = fir.coordinate_of %A, %i : (!fir.ref<f32>, index) -> !fir.ref<f32>
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
    OptionalAttr<UnitAttr>:$unordered
  );
  let results = (outs
    Variadic<AnyType>:$results
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "mlir::Value lowerBound, mlir::Value upperBound,"
              "mlir::Value step, bool unordered = false,"
              "ValueRange iterArgs = llvm::None,"
              "ArrayRef<NamedAttribute> attributes = {}">
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef unorderedAttrName() { return "unordered"; }

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

    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - 1;
    }
    /// Number of operands controlling the loop: lb, ub, step
    unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return getOperation()->getNumOperands() - getNumControlOperands();
    }

    /// Get the body of the loop
    mlir::Block *getBody() { return &region().front(); }

    void setUnordered() {
      getOperation()->setAttr(unorderedAttrName(),
                              mlir::UnitAttr::get(getContext()));
    }
  }];
}

def fir_WhereOp : region_Op<"if", [NoRegionArguments]> {
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
    SizedRegion<1>:$whereRegion,
    AnyRegion:$otherRegion
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value cond, bool withOtherRegion">,
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "TypeRange resultTypes, Value cond, bool withOtherRegion">
  ];

  let extraClassDeclaration = [{
    mlir::OpBuilder getWhereBodyBuilder() {
      assert(!whereRegion().empty() && "Unexpected empty 'where' region.");
      mlir::Block &body = whereRegion().front();
      return mlir::OpBuilder(&body, std::prev(body.end()));
    }
    mlir::OpBuilder getOtherBodyBuilder() {
      assert(!otherRegion().empty() && "Unexpected empty 'other' region.");
      mlir::Block &body = otherRegion().front();
      return mlir::OpBuilder(&body, std::prev(body.end()));
    }
  }];
}

def fir_IterWhileOp : region_Op<"iterate_while",
    [DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "DO loop with early exit condition";
  let description = [{
    This construct is useful for lowering implied-DO loops. It is very similar
    to `fir::LoopOp` with the addition that it requires a single loop-carried
    bool value that signals an early exit condition to the operation. A `true`
    disposition means the next loop iteration should proceed. A `false`
    indicates that the `fir.iterate_while` operation should terminate and
    return its iteration arguments.
  }];

  let arguments = (ins
    Index:$lowerBound,
    Index:$upperBound,
    Index:$step,
    I1:$iterateIn,
    Variadic<AnyType>:$initArgs
  );
  let results = (outs
    I1:$iterateResult,
    Variadic<AnyType>:$results
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "mlir::Value lowerBound, mlir::Value upperBound,"
	      "mlir::Value step, mlir::Value iterate,"
              "ValueRange iterArgs = llvm::None,"
              "ArrayRef<NamedAttribute> attributes = {}">
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

    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(mlir::Value step) { getOperation()->setOperand(2, step); }

    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - 1;
    }
    /// Number of operands controlling the loop
    unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return getOperation()->getNumOperands() - getNumControlOperands();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Procedure call operations
//===----------------------------------------------------------------------===//

def fir_CallOp : fir_Op<"call",
    [MemoryEffects<[MemAlloc, MemFree, MemRead, MemWrite]>]> {
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

  let extraClassDeclaration = [{
    static constexpr StringRef calleeAttrName() { return "callee"; }
  }];
}

def fir_DispatchOp : fir_Op<"dispatch",
    [MemoryEffects<[MemAlloc, MemFree, MemRead, MemWrite]>]> {
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
      result.addAttribute("method",
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
    result.addAttribute("fn_type", mlir::TypeAttr::get(calleeType));
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getAttr("method") << '(';
    p.printOperand(object());
    if (arg_operand_begin() != arg_operand_end()) {
      p << ", ";
      p.printOperands(args());
    }
    p << ')';
    p.printOptionalAttrDict(getAttrs(), {"fn_type", "method"});
    auto resTy{getResultTypes()};
    llvm::SmallVector<mlir::Type, 8> argTy(getOperandTypes());
    p << " : " << mlir::FunctionType::get(argTy, resTy, getContext());
  }];

  let extraClassDeclaration = [{
    mlir::FunctionType getFunctionType();
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin() + 1; }
    operand_iterator arg_operand_end() { return operand_end(); }
    llvm::StringRef passArgAttrName() { return "pass_arg_pos"; }
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

  let results = (outs fir_SequenceType);

  let parser = [{
    auto &builder = parser.getBuilder();
    mlir::Attribute val;
    mlir::NamedAttrList attrs;
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
        parser.parseColonType(type))
      return mlir::failure();
    if (!(type.isa<fir::CharacterType>() || type.isa<mlir::IntegerType>()))
      return parser.emitError(parser.getCurrentLocation(),
                              "must have character type");
    type = fir::SequenceType::get({sz.getInt()}, type);
    if (!type || parser.addTypesToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getValue() << '(';
    p << getSize().cast<mlir::IntegerAttr>().getValue() << ") : ";
    p.printType(getType().cast<fir::SequenceType>().getEleTy());
  }];

  let verifier = [{
    if (getSize().cast<mlir::IntegerAttr>().getValue().isNegative())
      return emitOpError("size must be non-negative");
    auto eleTy = getType().cast<fir::SequenceType>().getEleTy();
    if (!eleTy.isa<fir::CharacterType>())
      return emitOpError("must have !fir.char type");
    if (auto xl = getAttr(xlist())) {
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
    mlir::Attribute getSize() { return getAttr(size()); }
    // Get the string value of this character constant
    mlir::Attribute getValue() {
      if (auto attr = getAttr(value()))
        return attr;
      return getAttr(xlist());
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

def FirRealAttr : Attr<CPred<"$_self.isa<fir::RealAttr>()">, "FIR real attr"> {
  let storageType = [{ fir::RealAttr }];
  let returnType = [{ llvm::APFloat }];
}

def fir_ConstfOp : fir_Op<"constf", [NoSideEffect]> {
  let summary = "create a floating point constant";

  let description = [{
    A floating-point constant. This operation is to augment MLIR to be able
    to represent APFloat values that are not supported in the standard dialect.
  }];

  let arguments = (ins FirRealAttr:$constant);
  
  let results = (outs fir_RealType:$res);

  let assemblyFormat = "`(` $constant `)` attr-dict `:` type($res)";
  
  let verifier = [{
    if (!getType().isa<fir::RealType>())
      return emitOpError("must be a !fir.real type");
    return mlir::success();
  }];
}

class RealUnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_UnaryArithmeticOp<mnemonic, traits>,
      Arguments<(ins AnyRealLike:$operand)>;

def fir_NegfOp : RealUnaryArithmeticOp<"negf">;

class RealArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_ArithmeticOp<mnemonic, traits>,
      Arguments<(ins AnyRealLike:$lhs, AnyRealLike:$rhs)>;

def fir_AddfOp : RealArithmeticOp<"addf", [Commutative]> {
  let hasFolder = 1;
}
def fir_SubfOp : RealArithmeticOp<"subf"> {
  let hasFolder = 1;
}
def fir_MulfOp : RealArithmeticOp<"mulf", [Commutative]> {
  let hasFolder = 1;
}
def fir_DivfOp : RealArithmeticOp<"divf">;
def fir_ModfOp : RealArithmeticOp<"modf">;
// Pow is a builtin call and not a primitive

def fir_CmpfOp : fir_Op<"cmpf",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape]> {
  let summary = "floating-point comparison operator";

  let description = [{
    Extends the standard floating-point comparison to handle the extended
    floating-point types found in FIR.
  }];

  let arguments = (ins AnyRealLike:$lhs, AnyRealLike:$rhs);

  let results = (outs AnyLogicalLike);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, CmpFPredicate predicate,"
    "Value lhs, Value rhs", [{
      buildCmpFOp(builder, result, predicate, lhs, rhs);
  }]>];

  let parser = [{ return parseCmpfOp(parser, result); }];

  let printer = [{ printCmpfOp(p, *this); }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getPredicateAttrName() {
      return "predicate";
    }
    static CmpFPredicate getPredicateByName(llvm::StringRef name);

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<mlir::IntegerAttr>(
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
    auto f1 = getAttr(realAttrName()).cast<mlir::FloatAttr>();
    auto i1 = f1.getValue().bitcastToAPInt();
    p.getStream().write_hex(i1.getZExtValue());
    p << ", 0x";
    auto f2 = getAttr(imagAttrName()).cast<mlir::FloatAttr>();
    auto i2 = f2.getValue().bitcastToAPInt();
    p.getStream().write_hex(i2.getZExtValue());
    p << ") : ";
    p.printType(getType());
  }];

  let verifier = [{
    if (!getType().isa<fir::CplxType>())
      return emitOpError("must be a !fir.complex type");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef realAttrName() { return "real"; }
    static constexpr llvm::StringRef imagAttrName() { return "imaginary"; }

    mlir::Attribute getReal() { return getAttr(realAttrName()); }
    mlir::Attribute getImaginary() { return getAttr(imagAttrName()); }
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, CmpFPredicate predicate,"
    "Value lhs, Value rhs", [{
      buildCmpCOp(builder, result, predicate, lhs, rhs);
  }]>];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getPredicateAttrName() {
      return "predicate";
    }

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<mlir::IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];
}

// Other misc. operations

def fir_AddrOfOp : fir_OneResultOp<"address_of", [NoSideEffect]> {
  let summary = "convert a symbol to an SSA value";
  
  let description = [{
    Convert a symbol (a function or global reference) to an SSA-value to be
    used in other Operations.

    ```mlir
      %p = fir.address_of(@symbol) : !fir.ref<f64>
    ```
  }];

  let arguments = (ins SymbolRefAttr:$symbol);

  let results = (outs fir_ReferenceType:$resTy);

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
}

def FortranTypeAttr : Attr<And<[CPred<"$_self.isa<TypeAttr>()">,
    Or<[CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::CharacterType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::CplxType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::IntType>()">,
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
    p << getOperationName() << ' ' << getAttr("in_type");
    p.printOptionalAttrDict(getAttrs(), {"in_type"});
  }];

  let builders = [
    OpBuilder<"OpBuilder &, OperationState &result, mlir::TypeAttr inty">
  ];

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
      return getAttrOfType<mlir::TypeAttr>("in_type").getValue();
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
    p.printAttributeWithoutType(getAttr(symbolAttrName()));
    if (auto val = getValueOrNull())
      p << '(' << val << ')';
    if (getAttr(constantAttrName()))
      p << " constant";
    p << " : ";
    p.printType(getType());
    if (hasInitializationBody())
      p.printRegion(getOperation()->getRegion(0), /*printEntryBlockArgs=*/false,
                    /*printBlockTerminators=*/true);
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "StringRef name, Type type, ArrayRef<NamedAttribute> attrs = {}">,
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "StringRef name, bool isConstant, Type type,"
	      "ArrayRef<NamedAttribute> attrs = {}">,
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "StringRef name, Type type, StringAttr linkage = {},"
	      "ArrayRef<NamedAttribute> attrs = {}">,
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "StringRef name, bool isConstant, Type type,"
	      "StringAttr linkage = {},"
	      "ArrayRef<NamedAttribute> attrs = {}">,
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "StringRef name, Type type, Attribute initVal,"
	      "StringAttr linkage = {},"
	      "ArrayRef<NamedAttribute> attrs = {}">,
    OpBuilder<"mlir::OpBuilder &builder, OperationState &result,"
              "StringRef name, bool isConstant, Type type,"
	      "Attribute initVal, StringAttr linkage = {},"
	      "ArrayRef<NamedAttribute> attrs = {}">,
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef symbolAttrName() { return "symref"; }
    static constexpr llvm::StringRef constantAttrName() { return "constant"; }
    static constexpr llvm::StringRef initValAttrName() { return "initVal"; }
    static constexpr llvm::StringRef linkageAttrName() { return "linkName"; }
    static constexpr llvm::StringRef typeAttrName() { return "type"; }

    /// The printable type of the global
    mlir::Type getType() {
      return getAttrOfType<TypeAttr>(typeAttrName()).getValue();
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
    mlir::Region &getRegion() { return getOperation()->getRegion(0); }

    /// A GlobalOp has one block.
    mlir::Block &getBlock() { return getRegion().front(); }

    /// Determine if `linkage` is a supported keyword
    static mlir::ParseResult verifyValidLinkage(StringRef linkage);

    bool hasInitializationBody() {
      return (getOperation()->getNumRegions() == 1) && !getRegion().empty() &&
        !isa<fir::FirEndOp>(getBlock().front());
    }

    mlir::FlatSymbolRefAttr getSymbol() {
      return mlir::FlatSymbolRefAttr::get(getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName()).getValue(), getContext());
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
    p << getOperationName() << ' ' << getAttr(lenParamAttrName()) << ", "
      << getAttr(intAttrName());
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
    if (parser.parseOptionalRegion(*body, llvm::None, llvm::None))
      return mlir::failure();

    ensureTerminator(*body, parser.getBuilder(), result.location);
    return mlir::success();
  }];

  let printer = [{
    auto tableName = getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName()).getValue();
    p << getOperationName() << " @" << tableName;

    Region &body = getOperation()->getRegion(0);
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
    OpBuilder<"mlir::OpBuilder &builder, OperationState *result,"
              "StringRef name, Type type, ArrayRef<NamedAttribute> attrs = {}",
    [{
      result->addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                           builder.getStringAttr(name));
      result->addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Append a dispatch table entry to the table.
    void appendTableEntry(mlir::Operation *op);

    mlir::Region &getRegion() {
      return this->getOperation()->getRegion(0);
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
      if (parser.parseAttribute(methodAttr, methodAttrName(),
                                result.attributes))
        return mlir::failure();
    } else {
      result.addAttribute(methodAttrName(),
          parser.getBuilder().getStringAttr(methodName));
    }
    mlir::SymbolRefAttr calleeAttr;
    if (parser.parseComma() ||
        parser.parseAttribute(calleeAttr, procAttrName(), result.attributes))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getAttr(methodAttrName()) << ", "
      << getAttr(procAttrName());
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef methodAttrName() { return "method"; }
    static constexpr llvm::StringRef procAttrName() { return "proc"; }
  }];
}

#endif
