// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -Wripple -fenable-ripple -o - %s -DUSE_PRAGMA | FileCheck --implicit-check-not="warning:" %s
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -Wripple -fenable-ripple -o - %s -DUSE_CALL | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

// CHECK: test1
void test1(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);

  // CHECK: ripple.par.block.size = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.loop.iters = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.loop.iters = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.init = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.step = alloca i{{[0-9]+}}
  // We allocate i as part of ripple codegen since it's declared in the loop init
  // CHECK-NEXT: i = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.iv = alloca i{{[0-9]+}}
  // CHECK: br label %ripple.par.for.begin

  // CHECK: ripple.par.for.begin:
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.par.block.size
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.loop.iters
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.par.loop.iters
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.par.init
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.par.step
  // CHECK: store i{{[0-9]+}} 0, ptr %ripple.par.iv
  // CHECK: br label %for.cond

  // The condition is ripple_par_iv < ripple_par_loop_iters
  // CHECK: for.cond:
  // CHECK: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK: %[[RippleNumIters:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.loop.iters
  // CHECK: %[[BranchCond:[A-Za-z0-9_]+]] = icmp ult i{{[0-9]+}} %[[RippleIV]], %[[RippleNumIters]]
  // CHECK: br i1 %[[BranchCond]], label %for.body, label %for.end

  // CHECK: for.body:
  // Update the original loop induction var i
  // CHECK-NEXT: %[[RippleInit:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.init
  // CHECK-NEXT: %[[RippleStep:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.step
  // CHECK-NEXT: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK-NEXT: %[[Offset:[A-Za-z0-9_]+]] = mul i{{[0-9]+}} %[[RippleStep]], %[[RippleIV]]
  // CHECK-NEXT: %[[InitPlusOffset:[A-Za-z0-9_]+]] = add i{{[0-9]+}} %[[RippleInit]], %[[Offset]]
  // CHECK-NEXT: store i{{[0-9]+}} %[[InitPlusOffset]], ptr %i
  // CHECK: br label %for.inc

  // Increments ripple_par_iv by 1
  // CHECK: for.inc:
  // CHECK: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK: %[[RippleIVPlusOne:[A-Za-z0-9_]+]] = add i{{[0-9]+}} %[[RippleIV]], 1
  // CHECK: store i{{[0-9]+}} %[[RippleIVPlusOne]], ptr %ripple.par.iv
  // CHECK: br label %for.cond

  // Scalar precondition of the masked section
  // CHECK: for.end:

  // Update the IV
  // CHECK-NEXT: %[[RippleInit:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.init
  // CHECK-NEXT: %[[RippleStep:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.step
  // CHECK-NEXT: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK-NEXT: %[[Offset:[A-Za-z0-9_]+]] = mul i{{[0-9]+}} %[[RippleStep]], %[[RippleIV]]
  // CHECK-NEXT: %[[InitPlusOffset:[A-Za-z0-9_]+]] = add i{{[0-9]+}} %[[RippleInit]], %[[Offset]]
  // CHECK-NEXT: store i{{[0-9]+}} %[[InitPlusOffset]], ptr %{{[A-Za-z0-9_]+}}

  // Check if we executed all iterations, bypassing the remainder if true
  // CHECK-NEXT: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK-NEXT: %[[RippleBlockSize:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.block.size
  // CHECK-NEXT: %[[IVTimesBlock:[A-Za-z0-9_]+]] = mul i{{[0-9]+}} %[[RippleIV]], %[[RippleBlockSize]]
  // CHECK-NEXT: %[[LoopIters:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.loop.iters
  // CHECK-NEXT: %[[Cond:[A-Za-z0-9_]+]] = icmp ne i{{[0-9]+}} %[[IVTimesBlock]], %[[LoopIters]]
  // CHECK-NEXT: br i1 %[[Cond]], label %ripple.par.for.remainder.cond, label %ripple.par.for.end

  // CHECK: ripple.par.for.remainder.cond:
  // Update the original loop induction var
  // CHECK: %[[OriginalCond:[A-Za-z0-9_]+]] = icmp ule i{{[0-9]+}}
  // CHECK: br i1 %[[OriginalCond]], label %ripple.par.for.remainder.body, label %ripple.par.for.end

  // CHECK: ripple.par.for.remainder.body:
  // Update IV to the UB
  // We are in the masked region (case where the number of iteration is not a multiple of the parallel region),
  //   hence we compute the UB as-if we executed the loop sequentially (to get the real UB and not the next multiple of the parallel step),
  //   i.e., IV = LB + Step * NumIter
  // CHECK: %[[NParam:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %n
  // CHECK-NEXT: %[[NPlusOne:[A-Za-z0-9_]+]] = sub i{{[0-9]+}} %[[NParam]], -1
  // CHECK-NEXT: %[[NumLoopIters:[A-Za-z0-9_]+]] = udiv i{{[0-9]+}} %[[NPlusOne]], 1
  // CHECK-NEXT: %[[TotalStride:[A-Za-z0-9_]+]] = mul i{{[0-9]+}} %[[NumLoopIters]], 1
  // CHECK-NEXT: %[[UpperBound:[A-Za-z0-9_]+]] = add i{{[0-9]+}} 0, %[[TotalStride]]
  // CHECK-NEXT: store i{{[0-9]+}} %[[UpperBound]], ptr %i
  // CHECK-NEXT: br label %ripple.par.for.end

  // CHECK: ripple.par.for.end:

#ifdef USE_PRAGMA
  #pragma ripple parallel Block(BS) Dims(0, 1)
#elif USE_CALL
  ripple_parallel(BS, 0, 1)
#else
  #error "Should define USE_CALL or USE_PRAGMA to test this file"
#endif
  for (size_t i = 0; i <= n; i++)
    C[i] = A[i] + B[i];

}

// CHECK: test2
void test2(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);

  // CHECK: ripple.par.block.size = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.loop.iters = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.loop.iters = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.init = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.step = alloca i{{[0-9]+}}
  // We allocate i as part of ripple codegen since it's declared in the loop init
  // CHECK-NEXT: i = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.iv = alloca i{{[0-9]+}}
  // CHECK: br label %ripple.par.for.begin

  // CHECK: ripple.par.for.begin:
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.par.block.size
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.loop.iters
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.par.loop.iters
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.par.init
  // CHECK: store i{{[0-9]+}} %{{[0-9A-Za-z_]+}}, ptr %ripple.par.step
  // CHECK: store i{{[0-9]+}} 0, ptr %ripple.par.iv
  // CHECK: br label %for.cond

  // The condition is ripple_par_iv < ripple_par_loop_iters
  // CHECK: for.cond:
  // CHECK: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK: %[[RippleNumIters:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.loop.iters
  // CHECK: %[[BranchCond:[A-Za-z0-9_]+]] = icmp ult i{{[0-9]+}} %[[RippleIV]], %[[RippleNumIters]]
  // CHECK: br i1 %[[BranchCond]], label %for.body, label %for.end

  // CHECK: for.body:
  // Update the original loop induction var
  // CHECK: %[[RippleInit:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.init
  // CHECK: %[[RippleStep:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.step
  // CHECK: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK: %[[Offset:[A-Za-z0-9_]+]] = mul i{{[0-9]+}} %[[RippleStep]], %[[RippleIV]]
  // CHECK: %[[InitPlusOffset:[A-Za-z0-9_]+]] = add i{{[0-9]+}} %[[RippleInit]], %[[Offset]]
  // CHECK: store i{{[0-9]+}} %[[InitPlusOffset]], ptr %i
  // CHECK: br label %for.inc

  // Increments ripple_par_iv by 1
  // CHECK: for.inc:
  // CHECK: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK: %[[RippleIVPlusOne:[A-Za-z0-9_]+]] = add i{{[0-9]+}} %[[RippleIV]], 1
  // CHECK: store i{{[0-9]+}} %[[RippleIVPlusOne]], ptr %ripple.par.iv
  // CHECK: br label %for.cond

  // Scalar precondition of the masked section
  // CHECK: for.end:
  // Update IV to the UB
  // CHECK-NEXT: %[[RippleInit:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.init
  // CHECK-NEXT: %[[RippleStep:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.step
  // CHECK-NEXT: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK-NEXT: %[[Offset:[A-Za-z0-9_]+]] = mul i{{[0-9]+}} %[[RippleStep]], %[[RippleIV]]
  // CHECK-NEXT: %[[InitPlusOffset:[A-Za-z0-9_]+]] = add i{{[0-9]+}} %[[RippleInit]], %[[Offset]]
  // CHECK-NEXT: store i{{[0-9]+}} %[[InitPlusOffset]], ptr %{{[A-Za-z0-9_]+}}
  // CHECK-NEXT: br label %ripple.par.for.end

  // CHECK: ripple.par.for.end:

#ifdef USE_PRAGMA
  #pragma ripple parallel Block(BS) Dims(0, 1) NoRemainder
#elif USE_CALL
  ripple_parallel_full(BS, 0, 1)
#else
  #error "Should define USE_CALL or USE_PRAGMA to test this file"
#endif
  for (size_t i = 0; i < n; i += 2)
    C[i] = A[i] + B[i];

}

// CHECK: test3
void test3(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);

  // I is declared outside the loop
  // CHECK: i = alloca i{{[0-9]+}}
  // CHECK: ripple.par.block.size = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.loop.iters = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.loop.iters = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.init = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.step = alloca i{{[0-9]+}}
  // CHECK-NEXT: ripple.par.iv = alloca i{{[0-9]+}}
  // CHECK-NOT: i = alloca i{{[0-9]+}}
  // CHECK: br label %ripple.par.for.begin

  // CHECK: for.end:
  // Update IV to the UB
  // CHECK-NEXT: %[[RippleInit:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.init
  // CHECK-NEXT: %[[RippleStep:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.step
  // CHECK-NEXT: %[[RippleIV:[A-Za-z0-9_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
  // CHECK-NEXT: %[[Offset:[A-Za-z0-9_]+]] = mul i{{[0-9]+}} %[[RippleStep]], %[[RippleIV]]
  // CHECK-NEXT: %[[InitPlusOffset:[A-Za-z0-9_]+]] = add i{{[0-9]+}} %[[RippleInit]], %[[Offset]]
  // CHECK-NEXT: store i{{[0-9]+}} %[[InitPlusOffset]], ptr %{{[A-Za-z0-9_]+}}
  // CHECK-NEXT: br label %ripple.par.for.end

  // CHECK: ripple.par.for.end:

  size_t i;
#ifdef USE_PRAGMA
  #pragma ripple parallel Block(BS) Dims(0, 1) NoRemainder
#elif USE_CALL
  ripple_parallel_full(BS, 0, 1)
#else
  #error "Should define USE_CALL or USE_PRAGMA to test this file"
#endif
  for (i = 0; i < n; i++)
    C[i] = A[i] + B[i];

}
