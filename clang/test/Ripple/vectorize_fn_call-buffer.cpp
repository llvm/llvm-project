// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon-unknown-elf -fenable-ripple -emit-llvm %s -o - -O0 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-BUFFER

#include <ripple.h>

// Test case: calling non-void function when number of the vector elements is equal to the blocksize
extern "C" float inner_fn(float in, float m);

extern "C" void general_func_call_1(float in[32], float out[32], float factor) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t v = ripple_id(BS, 0);
  out[v] = inner_fn(in[v], factor);
  out[v] = in[v] + out[v];
}
// CHECK:               @general_func_call_1
// CHECK-BUFFER:        %[[OUTBUF:[0-9]+]] = alloca [32 x float]
// CHECK-BUFFER:        %[[INBUF:[0-9]+]] = alloca [32 x float]
// CHECK:               %[[FACTOR:[0-9]+]] = load float, ptr %factor.addr

// Pre-Header
// CHECK:               call.ripple.call.loop.pre:
// CHECK-NEXT:          br label %call.ripple.call.loop.header

// Header
// CHECK:               call.ripple.call.loop.header:
// CHECK:               %[[IndVar:[a-zA-Z0-9_.]+]] = phi i{{[0-9]+}} [ 0, %call.ripple.call.loop.pre ], [ %[[IncVar:[0-9]+]], %call.ripple.call.loop.body.continue.block ]
// CHECK-NEXT:          %[[ZeroInit:[a-zA-Z0-9_.]+]] = select <32 x i1> splat (i1 true), <32 x i8> splat (i8 1), <32 x i8> zeroinitializer
// CHECK:               %[[LoopCond:[0-9]+]] = icmp ult i{{[0-9]+}} %[[IndVar]], 32
// CHECK-NEXT:          br i1 %[[LoopCond]], label %call.ripple.call.loop.body, label %call.ripple.call.loop.end

// Loop
// CHECK:               call.ripple.call.loop.body:
// CHECK-BUFFER-NEXT:   %[[MaskVal:[a-zA-Z0-9_.]+]] = extractelement <32 x i8> %[[ZeroInit]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:          %[[MaskCond:[0-9]+]] = icmp eq i8 %[[MaskVal]], 1
// CHECK-NEXT:          br i1 %[[MaskCond]], label %call.ripple.call.loop.body.call.block, label %call.ripple.call.loop.body.continue.block

// CHECK:               call.ripple.call.loop.body.call.block:
// CHECK-BUFFER-NEXT:   %[[InScalAddr:[0-9]+]] = getelementptr [32 x float], ptr %[[INBUF]], i{{[0-9]+}} 0, i{{[0-9]+}} %[[IndVar]]
// CHECK-BUFFER-NEXT:   %[[InScalar:[0-9]+]] = load float, ptr %[[InScalAddr]]
// CHECK-NEXT:          %[[Ret:[0-9]+]] = call float @inner_fn(float %[[InScalar]], float %[[FACTOR]])
// CHECK-BUFFER-NEXT:   %[[OutBufAddr:[0-9]+]] = getelementptr [32 x float], ptr %[[OUTBUF]], i{{[0-9]+}} 0, i{{[0-9]+}} %[[IndVar]]
// CHECK-BUFFER-NEXT:   store float %[[Ret]], ptr %[[OutBufAddr]]
// CHECK-NEXT:          br label %call.ripple.call.loop.body.continue.block

// CHECK:               call.ripple.call.loop.body.continue.block
// CHECK:               %[[IncVar:[0-9]+]] = add i{{[0-9]+}} %[[IndVar]], 1
// CHECK-NEXT:          br label %call.ripple.call.loop.header

// CHECK:               call.ripple.call.loop.end:
// CHECK:               %[[AddArg0:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[AddArg1:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %{{[a-zA-Z0-9_.]+}} = fadd <32 x float> %[[AddArg0]], %[[AddArg1]]
// CHECK:               ret void

// Test case: calling void function when number of the vector elements is equal to the blocksize
extern "C" void inner_fn_void(float in, float m, float &out);

extern "C" void general_func_call_2(float in[32], float out[32], float factor) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t v = ripple_id(BS, 0);
  inner_fn_void(in[v], factor, out[v]);
  out[v] = in[v] + out[v];
}
// CHECK:               @general_func_call_2
// CHECK-BUFFER:        %[[OUTBUF:[0-9]+]] = alloca [32 x ptr]
// CHECK-BUFFER:        %[[INBUF:[0-9]+]] = alloca [32 x float]
// CHECK:               %[[FACTOR:[0-9]+]] = load float, ptr %factor.addr

// Pre-Header
// CHECK:       .ripple.call.loop.pre:
// CHECK-NEXT:  br label %.ripple.call.loop.header

// Header
// CHECK:             .ripple.call.loop.header:
// CHECK-NEXT:        %[[IndVar:[a-zA-Z0-9_.]+]] = phi i{{[0-9]+}} [ 0, %.ripple.call.loop.pre ], [ %[[IncVar:[0-9]+]], %.ripple.call.loop.body.continue.block ]
// CHECK-NEXT:        %[[ZeroInit:[a-zA-Z0-9_.]+]] = select <32 x i1> splat (i1 true), <32 x i8> splat (i8 1), <32 x i8> zeroinitializer
// CHECK:             %[[LoopCond:[0-9]+]] = icmp ult i{{[0-9]+}} %[[IndVar]], 32
// CHECK-NEXT:        br i1 %[[LoopCond]], label %.ripple.call.loop.body, label %.ripple.call.loop.end

// Loop
// CHECK:              .ripple.call.loop.body:
// CHECK-BUFFER-NEXT:   %[[MaskVal:[a-zA-Z0-9_.]+]] = extractelement <32 x i8> %[[ZeroInit]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:          %[[MaskCond:[0-9]+]] = icmp eq i8 %[[MaskVal]], 1
// CHECK-NEXT:          br i1 %[[MaskCond]], label %.ripple.call.loop.body.call.block, label %.ripple.call.loop.body.continue.block



// CHECK:                .ripple.call.loop.body.call.block:
// CHECK-BUFFER-NEXT:    %[[InScalAddr:[0-9]+]] = getelementptr [32 x float], ptr %[[INBUF]], i{{[0-9]+}} 0, i{{[0-9]+}} %[[IndVar]]
// CHECK-BUFFER-NEXT:    %[[InScalar:[0-9]+]] = load float, ptr %[[InScalAddr]]
// CHECK-BUFFER-NEXT:    %[[OutScalAddr:[0-9]+]] = getelementptr [32 x ptr], ptr %[[OUTBUF]], i{{[0-9]+}} 0, i{{[0-9]+}} %[[IndVar]]
// CHECK-BUFFER-NEXT:    %[[OutVal:[0-9]+]] = load ptr, ptr %[[OutScalAddr]]
// CHECK-NEXT:           call void @inner_fn_void(float %[[InScalar]], float %[[FACTOR]], ptr %[[OutVal]])
// CHECK-NEXT:           br label %.ripple.call.loop.body.continue.block

// CHECK:               .ripple.call.loop.body.continue.block:
// CHECK-NEXT:          %[[IncVar]] = add i{{[0-9]+}} %[[IndVar]], 1
// CHECK-NEXT:          br label %.ripple.call.loop.header

// CHECK:               .ripple.call.loop.end:
// CHECK:               %[[AddArg0:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[AddArg1:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %{{[a-zA-Z0-9_.]+}} = fadd <32 x float> %[[AddArg0]], %[[AddArg1]]
// CHECK:               ret void

// Test case: calling non-void function when number of vector elements is less than the blocksize
extern "C" float inner_fn(float in, float m);

extern "C" void general_func_call_3(float in[8], float out[8], float factor) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t v = ripple_id(BS, 0);
  if (v < 8)
    out[v] = inner_fn(in[v], factor);
  out[v] = in[v] + out[v];
}
// CHECK:               @general_func_call_3
// CHECK-BUFFER:        %[[OUTBUF:[0-9]+]] = alloca [32 x float]
// CHECK-BUFFER:        %[[INBUF:[0-9]+]] = alloca [32 x float]
// CHECK:               %[[CmpRipple:[a-zA-Z0-9_.]+]] = icmp ult <32 x i{{[0-9]+}}> %{{[a-zA-Z0-9_.]+}}, splat (i32 8)

// End-block
// CHECK:               if.end:
// CHECK:               %[[AddArg0:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[AddArg1:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %{{[a-zA-Z0-9_.]+}} = fadd <32 x float> %[[AddArg0]], %[[AddArg1]]
// CHECK:               ret void

// CHECK-BUFFER:        %[[FACTOR:[0-9]+]] = load float, ptr %factor.addr
// CHECK:               if.then.ripple.branch.clone.ripple.after.masked.load.ripple.after.masked.load:
// CHECK-NEXT:          %[[FACTORMaskedLoad:[a-zA-Z0-9_.]+]] = phi float [ %[[FACTOR]], %.ripple.branch.mask.apply{{[0-9]+}} ], [ poison, %if.then.ripple.branch.clone.ripple.after.masked.load ]

// Pre-Header
// CHECK:               call.ripple.call.loop.pre.ripple.branch.clone:
// CHECK-NEXT:          br label %call.ripple.call.loop.header.ripple.branch.clone

// Header
// CHECK:               call.ripple.call.loop.header.ripple.branch.clone:
// CHECK:               %[[IndVar:[a-zA-Z0-9_.]+]] = phi i{{[0-9]+}} [ 0, %call.ripple.call.loop.pre.ripple.branch.clone ], [ %[[IncVar:[0-9]+]], %call.ripple.call.loop.body.continue.block.ripple.branch.clone ]
// CHECK:               %[[MaskApply:[a-zA-Z0-9_.]+]] = and <32 x i1> splat (i1 true), %[[CmpRipple]]
// CHECK:               %[[ZeroInit:[a-zA-Z0-9_.]+]] = select <32 x i1> %[[MaskApply]], <32 x i8> splat (i8 1), <32 x i8> zeroinitializer
// CHECK-NEXT:          %[[LoopCond:[0-9]+]] = icmp ult i{{[0-9]+}} %[[IndVar]], 32
// CHECK-NEXT:          br i1 %[[LoopCond]], label %call.ripple.call.loop.body.ripple.branch.clone, label %call.ripple.call.loop.end

// Loop
// CHECK:               call.ripple.call.loop.body.ripple.branch.clone:
// CHECK-BUFFER-NEXT:   %[[MaskVal:[a-zA-Z0-9_.]+]] = extractelement <32 x i8> %[[ZeroInit]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:          %[[MaskCond:[0-9]+]] = icmp eq i8 %[[MaskVal]], 1
// CHECK-NEXT:          br i1 %[[MaskCond]], label %call.ripple.call.loop.body.call.block.ripple.branch.clone, label %call.ripple.call.loop.body.continue.block.ripple.branch.clone

// CHECK:               call.ripple.call.loop.body.call.block.ripple.branch.clone:
// CHECK-BUFFER-NEXT:   %[[InScalAddr:[0-9]+]] = getelementptr [32 x float], ptr %[[INBUF]], i{{[0-9]+}} 0, i{{[0-9]+}} %[[IndVar]]
// CHECK-BUFFER:        %[[InScalarLoad:[0-9]+]] = load float, ptr %[[InScalAddr]]
// CHECK-BUFFER:        %[[InScalar:[a-zA-Z0-9_.]+]] = phi float [ %[[InScalarLoad]], %.ripple.branch.mask.apply{{[0-9]+}} ], [ poison, %call.ripple.call.loop.body.call.block.ripple.branch.clone ]
// CHECK:               %[[Ret:[0-9]+]] = call float @inner_fn(float %[[InScalar]], float %[[FACTORMaskedLoad]])
// CHECK-BUFFER-NEXT:   %[[OutBufAddr:[0-9]+]] = getelementptr [32 x float], ptr %[[OUTBUF]], i{{[0-9]+}} 0, i{{[0-9]+}} %[[IndVar]]
// CHECK-BUFFER:        store float %[[Ret]], ptr %[[OutBufAddr]]
// CHECK:               br label %call.ripple.call.loop.body.continue.block.ripple.branch.clone

// CHECK:               call.ripple.call.loop.body.continue.block.ripple.branch.clone:
// CHECK:               %[[IncVar:[0-9]+]] = add i{{[0-9]+}} %[[IndVar]], 1
// CHECK-NEXT:          br label %call.ripple.call.loop.header


// Test case: calling void function when number of vector elements is less than the blocksize
extern "C" void inner_fn_void(float in, float m, float &out);

extern "C" void general_func_call_4(float in[32], float out[32], float factor) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t v = ripple_id(BS, 0);
  if (v < 8)
    inner_fn_void(in[v], factor, out[v]);
  out[v] = in[v] + out[v];
}
// CHECK:               @general_func_call_4
// CHECK-BUFFER:        %[[OUTBUF:[0-9]+]] = alloca [32 x ptr]
// CHECK-BUFFER:        %[[INBUF:[0-9]+]] = alloca [32 x float]
// CHECK:               %[[CmpRipple:[a-zA-Z0-9_.]+]] = icmp ult <32 x i{{[0-9]+}}> %{{[a-zA-Z0-9_.]+}}, splat (i32 8)

// End-block
// CHECK:               if.end:
// CHECK:               %[[AddArg0:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[AddArg1:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %{{[a-zA-Z0-9_.]+}} = fadd <32 x float> %[[AddArg0]], %[[AddArg1]]
// CHECK:               ret void


// CHECK:               %[[FACTOR:[0-9]+]] = load float, ptr %factor.addr
// CHECK:               if.then.ripple.branch.clone.ripple.after.masked.load.ripple.after.masked.load:
// CHECK-NEXT:          %[[FACTORMaskedLoad:[a-zA-Z0-9_.]+]] = phi float [ %[[FACTOR]], %.ripple.branch.mask.apply{{[0-9]+}} ], [ poison, %if.then.ripple.branch.clone.ripple.after.masked.load ]

// CHECK:               br label %.ripple.call.loop.pre.ripple.branch.clone


// Pre-Header
// CHECK:               .ripple.call.loop.pre.ripple.branch.clone:
// CHECK-NEXT:          br label %.ripple.call.loop.header.ripple.branch.clone

// Header
// CHECK:               .ripple.call.loop.header.ripple.branch.clone:
// CHECK:               %[[IndVar:[a-zA-Z0-9_.]+]] = phi i{{[0-9]+}} [ 0, %.ripple.call.loop.pre.ripple.branch.clone ], [ %[[IncVar:[0-9]+]], %.ripple.call.loop.body.continue.block.ripple.branch.clone ]
// CHECK:               %[[MaskApply:[a-zA-Z0-9_.]+]] = and <32 x i1> splat (i1 true), %[[CmpRipple]]
// CHECK:               %[[ZeroInit:[a-zA-Z0-9_.]+]] = select <32 x i1> %[[MaskApply]], <32 x i8> splat (i8 1), <32 x i8> zeroinitializer
// CHECK-NEXT:          %[[LoopCond:[0-9]+]] = icmp ult i{{[0-9]+}} %[[IndVar]], 32
// CHECK-NEXT:          br i1 %[[LoopCond]], label %.ripple.call.loop.body.ripple.branch.clone, label %.ripple.call.loop.end

// Loop
// CHECK:               .ripple.call.loop.body.ripple.branch.clone:
// CHECK-BUFFER-NEXT:   %[[MaskVal:[a-zA-Z0-9_.]+]] = extractelement <32 x i8> %[[ZeroInit]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:          %[[MaskCond:[0-9]+]] = icmp eq i8 %[[MaskVal]], 1
// CHECK-NEXT:          br i1 %[[MaskCond]], label %.ripple.call.loop.body.call.block.ripple.branch.clone, label %.ripple.call.loop.body.continue.block.ripple.branch.clone

// CHECK:               .ripple.call.loop.body.call.block.ripple.branch.clone:
// CHECK-BUFFER-NEXT:   %[[InScalAddr:[0-9]+]] = getelementptr [32 x float], ptr %[[INBUF]], i{{[0-9]+}} 0, i{{[0-9]+}} %[[IndVar]]
// CHECK-BUFFER:        %[[InScalarLoad:[0-9]+]] = load float, ptr %[[InScalAddr]]
// CHECK-BUFFER:        %[[InScalar:[a-zA-Z0-9_.]+]] = phi float [ %[[InScalarLoad]], %.ripple.branch.mask.apply{{[0-9]+}} ], [ poison, %.ripple.call.loop.body.call.block.ripple.branch.clone ]
// CHECK-BUFFER:        %[[OutScalAddr:[0-9]+]] = getelementptr [32 x ptr], ptr %[[OUTBUF]], i{{[0-9]+}} 0, i{{[0-9]+}} %[[IndVar]]
// CHECK-BUFFER:        %[[OutValLoad:[0-9]+]] = load ptr, ptr %[[OutScalAddr]]
// CHECK-BUFFER:        %[[OutVal:[a-zA-Z0-9_.]+]] = phi ptr [ %[[OutValLoad]], %.ripple.branch.mask.apply{{[0-9]+}} ], [ poison, %.ripple.call.loop.body.call.block.ripple.branch.clone.ripple.after.masked.load ]
// CHECK:               call void @inner_fn_void(float %[[InScalar]], float %[[FACTORMaskedLoad]], ptr %[[OutVal]])
// CHECK:               br label %.ripple.call.loop.body.continue.block.ripple.branch.clone

// CHECK:               .ripple.call.loop.body.continue.block.ripple.branch.clone:
// CHECK:               %[[IncVar:[0-9]+]] = add i{{[0-9]+}} %[[IndVar]], 1
// CHECK-NEXT:          br label %.ripple.call.loop.header
