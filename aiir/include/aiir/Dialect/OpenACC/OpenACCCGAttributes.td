//===- OpenACCCGAttributes.td - OpenACC codegen attributes *- tablegen -*-===//
//
// Part of the AIIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines OpenACC codegen attributes (GPU parallel dimensions used for
// privatization, barrier management, and loop work-sharing).
//
//===----------------------------------------------------------------------===//

#ifndef OPENACCCG_ATTRIBUTES
#define OPENACCCG_ATTRIBUTES

def OpenACC_GPUParallelDimAttr : OpenACC_Attr<"GPUParallelDim", "par_dim"> {
  let summary = "GPU parallel dimension for use in OpenACC parallelism assignment.";
  let description = [{
    Identifies a single GPU parallel dimension. Used for privatization scope,
    barrier placement, and loop work-sharing in OpenACC codegen.
  }];
  let parameters = (ins "::aiir::IntegerAttr":$value);
  let extraClassDeclaration = [{
    // GPU dimension predicates
    bool isSeq() const;
    bool isThreadX() const;
    bool isThreadY() const;
    bool isThreadZ() const;
    bool isBlockX() const;
    bool isBlockY() const;
    bool isBlockZ() const;
    bool isAnyThread() const;
    bool isAnyBlock() const;

    // Attribute creation from gpu::Processor
    static GPUParallelDimAttr get(::aiir::AIIRContext *, ::aiir::gpu::Processor);

    // Get the underlying gpu::Processor
    ::aiir::gpu::Processor getProcessor() const;

    // Factory functions for each dimension
    static GPUParallelDimAttr seqDim(::aiir::AIIRContext *);
    static GPUParallelDimAttr threadXDim(::aiir::AIIRContext *);
    static GPUParallelDimAttr threadYDim(::aiir::AIIRContext *);
    static GPUParallelDimAttr threadZDim(::aiir::AIIRContext *);
    static GPUParallelDimAttr blockXDim(::aiir::AIIRContext *);
    static GPUParallelDimAttr blockYDim(::aiir::AIIRContext *);
    static GPUParallelDimAttr blockZDim(::aiir::AIIRContext *);

    // Factory functions by index (0=X, 1=Y, 2=Z)
    static GPUParallelDimAttr threadDim(::aiir::AIIRContext *, unsigned index);
    static GPUParallelDimAttr blockDim(::aiir::AIIRContext *, unsigned index);

    // Ordering (for nesting validation: seq < thread < block)
    int getOrder() const;

    // Get the next higher/lower parallel dimension in the hierarchy
    GPUParallelDimAttr getOneHigher() const;
    GPUParallelDimAttr getOneLower() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def OpenACC_GPUParallelDimsAttr : OpenACC_Attr<"GPUParallelDims", "par_dims"> {
  let summary = "List of GPU parallel dimensions for use in OpenACC parallelism assignment.";
  let description = [{
    Ordered list of GPU parallel dimensions. Used for privatization scope, barrier
    placement, and loop work-sharing in OpenACC codegen.
  }];
  let parameters = (ins ArrayRefParameter<"::aiir::acc::GPUParallelDimAttr">:$array);
  let extraClassDeclaration = [{
    // Factory for sequential execution
    static GPUParallelDimsAttr seq(::aiir::AIIRContext *);

    // Check if this represents sequential execution (single seq dimension)
    bool isSeq() const;

    // Check if this represents parallel execution (not sequential)
    bool isParallel() const;

    // Check if this has multiple dimensions
    bool isMultiDim() const;

    // GPU dimension predicates
    bool hasAnyBlockLevel() const;
    bool hasOnlyBlockLevel() const;
    bool hasOnlyThreadYLevel() const;
    bool hasOnlyThreadXLevel() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

#endif // OPENACCCG_ATTRIBUTES
