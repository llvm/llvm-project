// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -fenable-ripple -S -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-REGISTER

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
// CHECK-REGISTER:      %[[InputVector:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[FACTOR:[0-9]+]] = load float, ptr %factor.addr

// Pre-Header
// CHECK:               call.ripple.call.loop.pre:
// CHECK-NEXT:          br label %call.ripple.call.loop.header

// Header
// CHECK:               call.ripple.call.loop.header:
// CHECK-REGISTER:      %[[ResPhi:[0-9]+]] = phi <32 x float> [ poison, %call.ripple.call.loop.pre ], [ %[[LoopResult:[0-9]+]], %call.ripple.call.loop.body.continue.block ]
// CHECK:               %[[IndVar:[a-zA-Z0-9_.]+]] = phi i{{[0-9]+}} [ 0, %call.ripple.call.loop.pre ], [ %[[IncVar:[0-9]+]], %call.ripple.call.loop.body.continue.block ]
// CHECK-NEXT:          %[[ZeroInit:[a-zA-Z0-9_.]+]] = select <32 x i1> splat (i1 true), <32 x i8> splat (i8 1), <32 x i8> zeroinitializer
// CHECK:               %[[LoopCond:[0-9]+]] = icmp ult i{{[0-9]+}} %[[IndVar]], 32
// CHECK-NEXT:          br i1 %[[LoopCond]], label %call.ripple.call.loop.body, label %call.ripple.call.loop.end

// Loop
// CHECK:               call.ripple.call.loop.body:
// CHECK-REGISTER-NEXT: %[[MaskVal:[a-zA-Z0-9_.]+]] = extractelement <32 x i8> %[[ZeroInit]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:          %[[MaskCond:[0-9]+]] = icmp eq i8 %[[MaskVal]], 1
// CHECK-NEXT:          br i1 %[[MaskCond]], label %call.ripple.call.loop.body.call.block, label %call.ripple.call.loop.body.continue.block

// CHECK:               call.ripple.call.loop.body.call.block:
// CHECK-REGISTER-NEXT: %[[InScalar:[0-9]+]] = extractelement <32 x float> %[[InputVector]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:          %[[Ret:[0-9]+]] = call float @inner_fn(float %[[InScalar]], float %[[FACTOR]])
// CHECK-REGISTER-NEXT: %[[CallResult:[0-9]+]] = insertelement <32 x float> %[[ResPhi]], float %[[Ret]], i{{[0-9]+}} %ripple.scalarcall.iterator
// CHECK-NEXT:          br label %call.ripple.call.loop.body.continue.block

// CHECK:               call.ripple.call.loop.body.continue.block
// CHECK-REGISTER-NEXT: %[[LoopResult]] = phi <32 x float> [ %[[CallResult]], %call.ripple.call.loop.body.call.block ], [ %[[ResPhi]], %call.ripple.call.loop.body ]
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
// CHECK-REGISTER:      %[[InputVector:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[FACTOR:[0-9]+]] = load float, ptr %factor.addr
// CHECK-REGISTER:      %[[OutAddr:[a-zA-Z0-9_.]+]] = load ptr, ptr %out.addr
// CHECK-REGISTER:      %[[OutInsert:[a-zA-Z0-9_.]+]] = insertelement <32 x ptr> poison, ptr %[[OutAddr]], i{{[0-9]+}} 0
// CHECK-REGISTER:      %[[OutSplat:[a-zA-Z0-9_.]+]] = shufflevector <32 x ptr> %[[OutInsert]], <32 x ptr> poison, <32 x i{{[0-9]+}}> zeroinitializer
// CHECK-REGISTER:      %[[OutAddresses:[a-zA-Z0-9_.]+]] = getelementptr inbounds float, <32 x ptr> %[[OutSplat]],

// Pre-Header
// CHECK:       .ripple.call.loop.pre:
// CHECK-NEXT:  br label %.ripple.call.loop.header

// Header
// CHECK:             .ripple.call.loop.header:
// CHECK-NEXT:          %[[IndVar:[a-zA-Z0-9_.]+]] = phi i{{[0-9]+}} [ 0, %.ripple.call.loop.pre ], [ %[[IncVar:[0-9]+]], %.ripple.call.loop.body.continue.block ]
// CHECK-NEXT:           %[[ZeroInit:[a-zA-Z0-9_.]+]] = select <32 x i1> splat (i1 true), <32 x i8> splat (i8 1), <32 x i8> zeroinitializer
// CHECK:             %[[LoopCond:[0-9]+]] = icmp ult i{{[0-9]+}} %[[IndVar]], 32
// CHECK-NEXT:          br i1 %[[LoopCond]], label %.ripple.call.loop.body, label %.ripple.call.loop.end

// Loop
// CHECK:                ripple.call.loop.body:
// CHECK-REGISTER-NEXT:  %[[MaskVal:[a-zA-Z0-9_.]+]] = extractelement <32 x i8> %[[ZeroInit]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:           %[[MaskCond:[0-9]+]] = icmp eq i8 %[[MaskVal]], 1
// CHECK-NEXT:           br i1 %[[MaskCond]], label %.ripple.call.loop.body.call.block, label %.ripple.call.loop.body.continue.block


// CHECK:                .ripple.call.loop.body.call.block:
// CHECK-REGISTER-NEXT:  %[[InScalar:[0-9]+]] = extractelement <32 x float> %[[InputVector]], i{{[0-9]+}} %[[IndVar]]
// CHECK-REGISTER-NEXT:  %[[OutVal:[0-9]+]] = extractelement <32 x ptr> %[[OutAddresses]], i{{[0-9]+}} %[[IndVar]]
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
// CHECK:               %[[CmpRipple:[a-zA-Z0-9_.]+]] = icmp ult <32 x i{{[0-9]+}}> %{{[a-zA-Z0-9_.]+}}, splat (i{{[0-9]+}} 8)

// End-block
// CHECK:               if.end:
// CHECK:               %[[AddArg0:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[AddArg1:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %{{[a-zA-Z0-9_.]+}} = fadd <32 x float> %[[AddArg0]], %[[AddArg1]]
// CHECK:               ret void

// CHECK-REGISTER:      %[[InputVector:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK-REGISTER:      %[[FACTOR:[0-9]+]] = load float, ptr %factor.addr
// CHECK:               if.then.ripple.branch.clone.ripple.after.masked.load.ripple.after.masked.load:
// CHECK-NEXT:          %[[FACTORMaskedLoad:[a-zA-Z0-9_.]+]] = phi float [ %[[FACTOR]], %.ripple.branch.mask.apply{{[0-9]+}} ], [ poison, %if.then.ripple.branch.clone.ripple.after.masked.load ]

// Pre-Header
// CHECK:               call.ripple.call.loop.pre.ripple.branch.clone:
// CHECK-NEXT:          br label %call.ripple.call.loop.header.ripple.branch.clone

// Header
// CHECK:               call.ripple.call.loop.header.ripple.branch.clone:
// CHECK-REGISTER:      %[[ResPhi:[0-9]+]] = phi <32 x float> [ poison, %call.ripple.call.loop.pre.ripple.branch.clone ], [ %[[LoopResult:[0-9]+]], %call.ripple.call.loop.body.continue.block.ripple.branch.clone ]
// CHECK:               %[[IndVar:[a-zA-Z0-9_.]+]] = phi i{{[0-9]+}} [ 0, %call.ripple.call.loop.pre.ripple.branch.clone ], [ %[[IncVar:[0-9]+]], %call.ripple.call.loop.body.continue.block.ripple.branch.clone ]
// CHECK:               %[[MaskApply:[a-zA-Z0-9_.]+]] = and <32 x i1> splat (i1 true), %[[CmpRipple]]
// CHECK:               %[[ZeroInit:[a-zA-Z0-9_.]+]] = select <32 x i1> %[[MaskApply]], <32 x i8> splat (i8 1), <32 x i8> zeroinitializer
// CHECK-NEXT:          %[[LoopCond:[0-9]+]] = icmp ult i{{[0-9]+}} %[[IndVar]], 32
// CHECK-NEXT:          br i1 %[[LoopCond]], label %call.ripple.call.loop.body.ripple.branch.clone, label %call.ripple.call.loop.end

// Loop
// CHECK:               call.ripple.call.loop.body.ripple.branch.clone:
// CHECK-REGISTER-NEXT: %[[MaskVal:[a-zA-Z0-9_.]+]] = extractelement <32 x i8> %[[ZeroInit]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:          %[[MaskCond:[0-9]+]] = icmp eq i8 %[[MaskVal]], 1
// CHECK-NEXT:          br i1 %[[MaskCond]], label %call.ripple.call.loop.body.call.block.ripple.branch.clone, label %call.ripple.call.loop.body.continue.block.ripple.branch.clone

// CHECK:               call.ripple.call.loop.body.call.block.ripple.branch.clone:
// CHECK-REGISTER-NEXT: %[[InScalar:[0-9]+]] = extractelement <32 x float> %[[InputVector]], i{{[0-9]+}} %[[IndVar]]
// CHECK:               %[[Ret:[0-9]+]] = call float @inner_fn(float %[[InScalar]], float %[[FACTORMaskedLoad]])
// CHECK-REGISTER-NEXT: %[[CallResult:[0-9]+]] = insertelement <32 x float> %[[ResPhi]], float %[[Ret]], i{{[0-9]+}} %ripple.scalarcall.iterator
// CHECK:               br label %call.ripple.call.loop.body.continue.block.ripple.branch.clone

// CHECK:               call.ripple.call.loop.body.continue.block.ripple.branch.clone:
// CHECK-REGISTER-NEXT: %[[LoopResult:[0-9]+]] = phi <32 x float> [ %[[CallResult]], %call.ripple.call.loop.body.call.block.ripple.branch.clone ], [ %[[ResPhi]], %call.ripple.call.loop.body.ripple.branch.clone ]
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
// CHECK:               %[[CmpRipple:[a-zA-Z0-9_.]+]] = icmp ult <32 x i{{[0-9]+}}> %{{[a-zA-Z0-9_.]+}}, splat (i{{[0-9]+}} 8)

// End-block
// CHECK:               if.end:
// CHECK:               %[[AddArg0:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[AddArg1:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %{{[a-zA-Z0-9_.]+}} = fadd <32 x float> %[[AddArg0]], %[[AddArg1]]
// CHECK:               ret void


// CHECK-REGISTER:      %[[InputVector:[0-9]+]] = call <32 x float> @llvm.masked.gather.v32f32.v32p0
// CHECK:               %[[FACTOR:[0-9]+]] = load float, ptr %factor.addr
// CHECK:               if.then.ripple.branch.clone.ripple.after.masked.load.ripple.after.masked.load:
// CHECK-NEXT:          %[[FACTORMaskedLoad:[a-zA-Z0-9_.]+]] = phi float [ %[[FACTOR]], %.ripple.branch.mask.apply{{[0-9]+}} ], [ poison, %if.then.ripple.branch.clone.ripple.after.masked.load ]

// CHECK-REGISTER:      %[[OutAddr:[a-zA-Z0-9_.]+]] = load ptr, ptr %out.addr
// CHECK-REGISTER:      %[[OutValPhi:[a-zA-Z0-9_.]+]] = phi ptr [ %[[OutAddr]], %.ripple.LS.instance7.ripple.branch.clone.ripple.branch.mask.apply ], [ poison, %if.then.ripple.branch.clone.ripple.after.masked.load.ripple.after.masked.load ]
// CHECK-REGISTER-NEXT: %[[OutInsert:[a-zA-Z0-9_.]+]] = insertelement <32 x ptr> poison, ptr %[[OutValPhi]], i{{[0-9]+}} 0
// CHECK-REGISTER-NEXT: %[[OutSplat:[a-zA-Z0-9_.]+]] = shufflevector <32 x ptr> %[[OutInsert]], <32 x ptr> poison, <32 x i{{[0-9]+}}> zeroinitializer
// CHECK-REGISTER:      %[[OutAddresses:[a-zA-Z0-9_.]+]] = getelementptr inbounds float, <32 x ptr> %[[OutSplat]]
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
// CHECK-REGISTER-NEXT: %[[MaskVal:[a-zA-Z0-9_.]+]] = extractelement <32 x i8> %[[ZeroInit]], i{{[0-9]+}} %[[IndVar]]
// CHECK-NEXT:          %[[MaskCond:[0-9]+]] = icmp eq i8 %[[MaskVal]], 1
// CHECK-NEXT:          br i1 %[[MaskCond]], label %.ripple.call.loop.body.call.block.ripple.branch.clone, label %.ripple.call.loop.body.continue.block.ripple.branch.clone


// CHECK:               .ripple.call.loop.body.call.block.ripple.branch.clone:
// CHECK-REGISTER-NEXT: %[[InScalar:[0-9]+]] = extractelement <32 x float> %[[InputVector]], i{{[0-9]+}} %[[IndVar]]
// CHECK-REGISTER-NEXT: %[[OutVal:[0-9]+]] = extractelement <32 x ptr> %[[OutAddresses]], i{{[0-9]+}} %[[IndVar]]
// CHECK:               call void @inner_fn_void(float %[[InScalar]], float %[[FACTORMaskedLoad]], ptr %[[OutVal]])
// CHECK:               br label %.ripple.call.loop.body.continue.block.ripple.branch.clone

// CHECK:               .ripple.call.loop.body.continue.block.ripple.branch.clone:
// CHECK:               %[[IncVar:[0-9]+]] = add i{{[0-9]+}} %[[IndVar]], 1
// CHECK-NEXT:          br label %.ripple.call.loop.header
