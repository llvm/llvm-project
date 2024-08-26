//===-- CUFOps.td - CUF operation definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the CUF dialect operations
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_CUF_CUF_OPS
#define FORTRAN_DIALECT_CUF_CUF_OPS

include "flang/Optimizer/Dialect/CUF/CUFDialect.td"
include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.td"
include "flang/Optimizer/Dialect/FIRTypes.td"
include "flang/Optimizer/Dialect/FIRAttr.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/BuiltinAttributes.td"

class cuf_Op<string mnemonic, list<Trait> traits>
    : Op<CUFDialect, mnemonic, traits>;

def cuf_AllocOp : cuf_Op<"alloc", [AttrSizedOperandSegments,
    MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate an object on device";

  let description = [{
    This is a drop in replacement for fir.alloca and fir.allocmem for device
    object. Any device, managed or unified object declared in an host
    subprogram needs to be allocated in the device memory through runtime calls.
    The cuf.alloc is an abstraction to the runtime calls and works together
    with cuf.free.
  }];

  let arguments = (ins
    TypeAttr:$in_type,
    OptionalAttr<StrAttr>:$uniq_name,
    OptionalAttr<StrAttr>:$bindc_name,
    Variadic<AnyIntegerType>:$typeparams,
    Variadic<AnyIntegerType>:$shape,
    cuf_DataAttributeAttr:$data_attr
  );

  let results = (outs fir_ReferenceType:$ptr);

  let assemblyFormat = [{
    $in_type (`(` $typeparams^ `:` type($typeparams) `)`)?
        (`,` $shape^ `:` type($shape) )?  attr-dict `->` qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$inType, "llvm::StringRef":$uniqName,
      "llvm::StringRef":$bindcName,
      "cuf::DataAttributeAttr":$cudaAttr,
      CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"mlir::ValueRange", "{}">:$shape,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>];

  let hasVerifier = 1;
}

def cuf_FreeOp : cuf_Op<"free", [MemoryEffects<[MemFree]>]> {
  let summary = "Free a device allocated object";

  let description = [{
    The cuf.free operation frees the memory allocated by cuf.alloc.
    This is used for non-allocatable device, managed and unified device
    variables declare in host subprogram.
  }];

  let arguments = (ins
    Arg<AnyReferenceLike, "", [MemFree]>:$devptr,
    cuf_DataAttributeAttr:$data_attr
  );

  let assemblyFormat = "$devptr `:` qualified(type($devptr)) attr-dict";

  let hasVerifier = 1;
}

def cuf_AllocateOp : cuf_Op<"allocate", [AttrSizedOperandSegments,
    MemoryEffects<[MemAlloc<DefaultResource>]>]> {
  let summary = "Perform the device allocation of data of an allocatable";

  let description = [{
    The cuf.allocate operation performs the allocation on the device
    of the data of an allocatable. The descriptor passed to the operation
    is initialized before with the standard flang runtime calls.
  }];

  let arguments = (ins Arg<fir_ReferenceType, "", [MemRead, MemWrite]>:$box,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg,
                       Optional<AnyIntegerType>:$stream,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$pinned,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemRead]>:$source,
                       cuf_DataAttributeAttr:$data_attr,
                       UnitAttr:$hasStat);

  let results = (outs AnyIntegerType:$stat);

  let assemblyFormat = [{
    $box `:` qualified(type($box))
    ( `source` `(` $source^ `:` qualified(type($source) )`)` )?
    ( `errmsg` `(` $errmsg^ `:` type($errmsg) `)` )?
    ( `stream` `(` $stream^ `:` type($stream) `)` )?
    ( `pinned` `(` $pinned^ `:` type($pinned) `)` )?
    attr-dict `->` type($stat)
  }];

  let hasVerifier = 1;
}

def cuf_DeallocateOp : cuf_Op<"deallocate",
    [MemoryEffects<[MemFree<DefaultResource>]>]> {
  let summary = "Perform the device deallocation of data of an allocatable";

  let description = [{
    The cuf.deallocate operation performs the deallocation on the device
    of the data of an allocatable.
  }];

  let arguments = (ins Arg<fir_ReferenceType, "", [MemRead, MemWrite]>:$box,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg,
                       cuf_DataAttributeAttr:$data_attr,
                       UnitAttr:$hasStat);

  let results = (outs AnyIntegerType:$stat);

  let assemblyFormat = [{
    $box `:` qualified(type($box))
    ( `errmsg` `(` $errmsg^ `:` type($errmsg) `)` )?
    attr-dict `->` type($stat)
  }];

  let hasVerifier = 1;
}

def cuf_DataTransferOp : cuf_Op<"data_transfer", []> {
  let summary = "Represent a data transfer between host and device memory";

  let description = [{
    CUDA Fortran allows data transfer to be done via intrinsic assignment
    between a host and a device variable. This operation is used to materialized
    the data transfer between the lhs and rhs memory references.
    The kind of transfer is specified in the attribute. 

    ```
      adev = a ! transfer host to device
      a = adev ! transfer device to host
      bdev = adev ! transfer device to device
    ```
    
    When the data transfer is done on data hold by descriptors, the LHS data
    hold by the descriptor are updated. When required, the LHS decriptor is also
    updated.
  }];

  let arguments = (ins Arg<AnyType, "", [MemRead]>:$src,
                       Arg<AnyRefOrBoxType, "", [MemWrite]>:$dst,
                       Optional<fir_ShapeType>:$shape,
                       cuf_DataTransferKindAttr:$transfer_kind);

  let assemblyFormat = [{
    $src `to` $dst (`,` $shape^ `:` type($shape) )? attr-dict `:` type($src) `,` type($dst)
  }];

  let hasVerifier = 1;
}

def cuf_KernelLaunchOp : cuf_Op<"kernel_launch", [CallOpInterface,
    AttrSizedOperandSegments]> {
  let summary = "call CUDA kernel";

  let description = [{
    Launch a CUDA kernel from the host.

    ```
      // launch simple kernel with no arguments. bytes and stream value are
      // optional in the chevron notation.
      cuf.kernel_launch @kernel<<<%gx, %gy, %gz, %bx, %by, %bz>>>()
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$callee,
    I32:$grid_x,
    I32:$grid_y,
    I32:$grid_z,
    I32:$block_x,
    I32:$block_y,
    I32:$block_z,
    Optional<I32>:$bytes,
    Optional<I32>:$stream,
    Variadic<AnyType>:$args
  );

  let assemblyFormat = [{
    $callee `<` `<` `<` $grid_x `,` $grid_y `,` $grid_z `,`$block_x `,`
        $block_y `,` $block_z ( `,` $bytes^ ( `,` $stream^ )? )? `>` `>` `>`
        `` `(` $args `)` ( `:` `(` type($args)^ `)` )? attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }

    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
    mlir::FunctionType getFunctionType();

    unsigned getNbNoArgOperand() {
      unsigned nbNoArgOperand = 5; // grids and blocks values are always present.
      if (getBytes()) ++nbNoArgOperand;
      if (getStream()) ++nbNoArgOperand;
      return nbNoArgOperand;
    }

    operand_range getArgOperands() {
      return {operand_begin() + getNbNoArgOperand(), operand_end()};
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return mlir::MutableOperandRange(
          *this, getNbNoArgOperand(), getArgs().size() - 1);
    }
  }];
}

def cuf_KernelOp : cuf_Op<"kernel", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {

  let description = [{
    Represent the CUDA Fortran kernel directive. The operation is a loop like
    operation that represents the iteration range of the embedded loop nest.

    When grid or block variadic operands are empty, a `*` only syntax was used
    in the Fortran code.
    If the `*` is mixed with values for either grid or block, these are
    represented by a 0 constant value.
  }];

  let arguments = (ins
    Variadic<I32>:$grid, // empty means `*`
    Variadic<I32>:$block, // empty means `*`
    Optional<I32>:$stream,
    Variadic<Index>:$lowerbound,
    Variadic<Index>:$upperbound,
    Variadic<Index>:$step,
    OptionalAttr<I64Attr>:$n,
    Variadic<AnyType>:$reduceOperands,
    OptionalAttr<ArrayAttr>:$reduceAttrs
  );

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `<` `<` `<` custom<CUFKernelValues>($grid, type($grid)) `,` 
                custom<CUFKernelValues>($block, type($block))
        ( `,` `stream` `=` $stream^ )? `>` `>` `>`
        ( `reduce` `(` $reduceOperands^ `:` type($reduceOperands) `:` $reduceAttrs `)` )?
        custom<CUFKernelLoopControl>($region, $lowerbound, type($lowerbound),
            $upperbound, type($upperbound), $step, type($step))
        attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get Number of variadic operands
    unsigned getNumOperands(unsigned idx) {
      auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
        getOperandSegmentSizeAttr());
      return static_cast<unsigned>(segments[idx]);
    }
    // Get Number of reduction operands
    unsigned getNumReduceOperands() {
      return getNumOperands(7);
    }
    /// Does the operation hold operands for reduction variables
    bool hasReduceOperands() {
      return getNumReduceOperands() > 0;
    }
  }];

  let hasVerifier = 1;
}

#endif // FORTRAN_DIALECT_CUF_CUF_OPS
