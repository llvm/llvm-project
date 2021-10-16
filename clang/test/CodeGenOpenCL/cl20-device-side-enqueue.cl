// RUN: %clang_cc1 -disable-noundef-analysis %s -cl-std=CL2.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "spir-unknown-unknown" | FileCheck %s --check-prefix=COMMON --check-prefix=B32
// RUN: %clang_cc1 -disable-noundef-analysis %s -cl-std=CL2.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "spir64-unknown-unknown" | FileCheck %s --check-prefix=COMMON --check-prefix=B64
// RUN: %clang_cc1 -disable-noundef-analysis %s -cl-std=CL2.0 -ffake-address-space-map -O1 -emit-llvm -o - -triple "spir64-unknown-unknown" | FileCheck %s --check-prefix=CHECK-LIFETIMES

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

typedef void (^bl_t)(local void *);
typedef struct {int a;} ndrange_t;

// COMMON: %struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }

// For a block global variable, first emit the block literal as a global variable, then emit the block variable itself.
// COMMON: [[BL_GLOBAL:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* [[INV_G:@[^ ]+]] to i8*) to i8 addrspace(4)*) }
// COMMON: @block_G ={{.*}} addrspace(1) constant %struct.__opencl_block_literal_generic addrspace(4)* addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BL_GLOBAL]] to %struct.__opencl_block_literal_generic addrspace(1)*) to %struct.__opencl_block_literal_generic addrspace(4)*)

// For anonymous blocks without captures, emit block literals as global variable.
// COMMON: [[BLG1:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG2:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG3:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG4:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG5:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG6:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*, i8 addrspace(3)*, i8 addrspace(3)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG7:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG8:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* [[INVG8:@[^ ]+]] to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG9:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* [[INVG9:@[^ ]+]] to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG10:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }
// COMMON: [[BLG11:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* {{@[^ ]+}} to i8*) to i8 addrspace(4)*) }

// Emits block literal [[BL_GLOBAL]], invoke function [[INV_G]] and global block variable @block_G
// COMMON: define internal spir_func void [[INV_G]](i8 addrspace(4)* %{{.*}}, i8 addrspace(3)* %{{.*}})
const bl_t block_G = (bl_t) ^ (local void *a) {};

void callee(int id, __global int *out) {
  out[id] = id;
}

// COMMON-LABEL: define{{.*}} spir_kernel void @device_side_enqueue(i32 addrspace(1)* %{{.*}}, i32 addrspace(1)* %b, i32 %i)
kernel void device_side_enqueue(global int *a, global int *b, int i) {
  // COMMON: %default_queue = alloca %opencl.queue_t*
  queue_t default_queue;
  // COMMON: %flags = alloca i32
  unsigned flags = 0;
  // COMMON: %ndrange = alloca %struct.ndrange_t
  ndrange_t ndrange;
  // COMMON: %clk_event = alloca %opencl.clk_event_t*
  clk_event_t clk_event;
  // COMMON: %event_wait_list = alloca %opencl.clk_event_t*
  clk_event_t event_wait_list;
  // COMMON: %event_wait_list2 = alloca [1 x %opencl.clk_event_t*]
  clk_event_t event_wait_list2[] = {clk_event};

  // COMMON: [[NDR:%[a-z0-9]+]] = alloca %struct.ndrange_t, align 4

  // B32: %[[BLOCK_SIZES1:.*]] = alloca [1 x i32]
  // B64: %[[BLOCK_SIZES1:.*]] = alloca [1 x i64]
  // CHECK-LIFETIMES: %[[BLOCK_SIZES1:.*]] = alloca [1 x i64]
  // B32: %[[BLOCK_SIZES2:.*]] = alloca [1 x i32]
  // B64: %[[BLOCK_SIZES2:.*]] = alloca [1 x i64]
  // CHECK-LIFETIMES: %[[BLOCK_SIZES2:.*]] = alloca [1 x i64]
  // B32: %[[BLOCK_SIZES3:.*]] = alloca [1 x i32]
  // B64: %[[BLOCK_SIZES3:.*]] = alloca [1 x i64]
  // CHECK-LIFETIMES: %[[BLOCK_SIZES3:.*]] = alloca [1 x i64]
  // B32: %[[BLOCK_SIZES4:.*]] = alloca [1 x i32]
  // B64: %[[BLOCK_SIZES4:.*]] = alloca [1 x i64]
  // CHECK-LIFETIMES: %[[BLOCK_SIZES4:.*]] = alloca [1 x i64]
  // B32: %[[BLOCK_SIZES5:.*]] = alloca [1 x i32]
  // B64: %[[BLOCK_SIZES5:.*]] = alloca [1 x i64]
  // CHECK-LIFETIMES: %[[BLOCK_SIZES5:.*]] = alloca [1 x i64]
  // B32: %[[BLOCK_SIZES6:.*]] = alloca [3 x i32]
  // B64: %[[BLOCK_SIZES6:.*]] = alloca [3 x i64]
  // CHECK-LIFETIMES: %[[BLOCK_SIZES6:.*]] = alloca [3 x i64]
  // B32: %[[BLOCK_SIZES7:.*]] = alloca [1 x i32]
  // B64: %[[BLOCK_SIZES7:.*]] = alloca [1 x i64]
  // CHECK-LIFETIMES: %[[BLOCK_SIZES7:.*]] = alloca [1 x i64]

  // Emits block literal on stack and block kernel [[INVLK1]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: store i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* [[INVL1:@__device_side_enqueue_block_invoke[^ ]*]] to i8*) to i8 addrspace(4)*), i8 addrspace(4)** %block.invoke
  // B32: [[BL:%[0-9]+]] = bitcast <{ i32, i32, i8 addrspace(4)*, i32 addrspace(1)*, i32, i32 addrspace(1)* }>* %block to %struct.__opencl_block_literal_generic*
  // B64: [[BL:%[0-9]+]] = bitcast <{ i32, i32, i8 addrspace(4)*, i32 addrspace(1)*, i32 addrspace(1)*, i32 }>* %block to %struct.__opencl_block_literal_generic*
  // COMMON: [[BL_I8:%[0-9]+]] ={{.*}} addrspacecast %struct.__opencl_block_literal_generic* [[BL]] to i8 addrspace(4)*
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_basic(
  // COMMON-SAME: %opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %struct.ndrange_t* byval(%struct.ndrange_t) [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVLK1:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(void) {
                   a[i] = b[i];
                 });

  // Emits block literal on stack and block kernel [[INVLK2]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[WAIT_EVNT:%[0-9]+]] ={{.*}} addrspacecast %opencl.clk_event_t{{.*}}** %event_wait_list to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // COMMON: [[EVNT:%[0-9]+]] ={{.*}} addrspacecast %opencl.clk_event_t{{.*}}** %clk_event to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // COMMON: store i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* [[INVL2:@__device_side_enqueue_block_invoke[^ ]*]] to i8*) to i8 addrspace(4)*), i8 addrspace(4)** %block.invoke
  // COMMON: [[BL:%[0-9]+]] = bitcast <{ i32, i32, i8 addrspace(4)*, i32{{.*}}, i32{{.*}}, i32{{.*}} }>* %block4 to %struct.__opencl_block_literal_generic*
  // COMMON: [[BL_I8:%[0-9]+]] ={{.*}} addrspacecast %struct.__opencl_block_literal_generic* [[BL]] to i8 addrspace(4)*
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_basic_events
  // COMMON-SAME: (%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]],  %struct.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t{{.*}}* addrspace(4)* [[WAIT_EVNT]], %opencl.clk_event_t{{.*}}* addrspace(4)* [[EVNT]],
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVLK2:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange, 2, &event_wait_list, &clk_event,
                 ^(void) {
                   a[i] = b[i];
                 });

  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_basic_events
  // COMMON-SAME: (%opencl.queue_t{{.*}}* {{%[0-9]+}}, i32 {{%[0-9]+}}, %struct.ndrange_t* {{.*}}, i32 1, %opencl.clk_event_t{{.*}}* addrspace(4)* null, %opencl.clk_event_t{{.*}}* addrspace(4)* null,
  enqueue_kernel(default_queue, flags, ndrange, 1, 0, 0,
                 ^(void) {
                   return;
                 });

  // Emits global block literal [[BLG1]] and block kernel [[INVGK1]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK-LIFETIMES: [[LIFETIME_PTR:%[0-9]+]] = bitcast [1 x i64]* %[[BLOCK_SIZES1]] to i8*
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // CHECK-LIFETIMES-NEXT: getelementptr inbounds [1 x i64], [1 x i64]* %[[BLOCK_SIZES1]], i64 0, i64 0
  // CHECK-LIFETIMES-LABEL: call spir_func i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], [1 x i32]* %[[BLOCK_SIZES1]], i32 0, i32 0
  // B32: store i32 256, i32* %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], [1 x i64]* %[[BLOCK_SIZES1]], i32 0, i32 0
  // B64: store i64 256, i64* %[[TMP]], align 8
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_varargs(
  // COMMON-SAME: %opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %struct.ndrange_t* [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK1:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG1]] to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1,
  // B32-SAME: i32* %[[TMP]])
  // B64-SAME: i64* %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 256);

  char c;
  // Emits global block literal [[BLG2]] and block kernel [[INVGK2]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK-LIFETIMES: [[LIFETIME_PTR:%[0-9]+]] = bitcast [1 x i64]* %[[BLOCK_SIZES2]] to i8*
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // CHECK-LIFETIMES-NEXT: getelementptr inbounds [1 x i64], [1 x i64]* %[[BLOCK_SIZES2]], i64 0, i64 0
  // CHECK-LIFETIMES-LABEL: call spir_func i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], [1 x i32]* %[[BLOCK_SIZES2]], i32 0, i32 0
  // B32: store i32 %{{.*}}, i32* %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], [1 x i64]* %[[BLOCK_SIZES2]], i32 0, i32 0
  // B64: store i64 %{{.*}}, i64* %[[TMP]], align 8
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_varargs(
  // COMMON-SAME: %opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %struct.ndrange_t* [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK2:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG2]] to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1,
  // B32-SAME: i32* %[[TMP]])
  // B64-SAME: i64* %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 c);

  // Emits global block literal [[BLG3]] and block kernel [[INVGK3]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x %opencl.clk_event_t*], [1 x %opencl.clk_event_t*]* %event_wait_list2, i{{32|64}} 0, i{{32|64}} 0
  // COMMON: [[WAIT_EVNT:%[0-9]+]] ={{.*}} addrspacecast %opencl.clk_event_t{{.*}}** [[AD]] to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // COMMON: [[EVNT:%[0-9]+]]  ={{.*}} addrspacecast %opencl.clk_event_t{{.*}}** %clk_event to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // CHECK-LIFETIMES: [[LIFETIME_PTR:%[0-9]+]] = bitcast [1 x i64]* %[[BLOCK_SIZES3]] to i8*
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // CHECK-LIFETIMES-NEXT: getelementptr inbounds [1 x i64], [1 x i64]* %[[BLOCK_SIZES3]], i64 0, i64 0
  // CHECK-LIFETIMES-LABEL: call spir_func i32 @__enqueue_kernel_events_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], [1 x i32]* %[[BLOCK_SIZES3]], i32 0, i32 0
  // B32: store i32 256, i32* %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], [1 x i64]* %[[BLOCK_SIZES3]], i32 0, i32 0
  // B64: store i64 256, i64* %[[TMP]], align 8
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_events_varargs
  // COMMON-SAME: (%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]],  %struct.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t{{.*}} [[WAIT_EVNT]], %opencl.clk_event_t{{.*}} [[EVNT]],
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK3:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG3]] to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1,
  // B32-SAME: i32* %[[TMP]])
  // B64-SAME: i64* %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 256);

  // Emits global block literal [[BLG4]] and block kernel [[INVGK4]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x %opencl.clk_event_t*], [1 x %opencl.clk_event_t*]* %event_wait_list2, i{{32|64}} 0, i{{32|64}} 0
  // COMMON: [[WAIT_EVNT:%[0-9]+]] ={{.*}} addrspacecast %opencl.clk_event_t{{.*}}** [[AD]] to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // COMMON: [[EVNT:%[0-9]+]]  ={{.*}} addrspacecast %opencl.clk_event_t{{.*}}** %clk_event to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // CHECK-LIFETIMES: [[LIFETIME_PTR:%[0-9]+]] = bitcast [1 x i64]* %[[BLOCK_SIZES4]] to i8*
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // CHECK-LIFETIMES-NEXT: getelementptr inbounds [1 x i64], [1 x i64]* %[[BLOCK_SIZES4]], i64 0, i64 0
  // CHECK-LIFETIMES-LABEL: call spir_func i32 @__enqueue_kernel_events_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], [1 x i32]* %[[BLOCK_SIZES4]], i32 0, i32 0
  // B32: store i32 %{{.*}}, i32* %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], [1 x i64]* %[[BLOCK_SIZES4]], i32 0, i32 0
  // B64: store i64 %{{.*}}, i64* %[[TMP]], align 8
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_events_varargs
  // COMMON-SAME: (%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]],  %struct.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t{{.*}}* addrspace(4)* [[WAIT_EVNT]], %opencl.clk_event_t{{.*}}* addrspace(4)* [[EVNT]],
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK4:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG4]] to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1,
  // B32-SAME: i32* %[[TMP]])
  // B64-SAME: i64* %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 c);

  long l;
  // Emits global block literal [[BLG5]] and block kernel [[INVGK5]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK-LIFETIMES: [[LIFETIME_PTR:%[0-9]+]] = bitcast [1 x i64]* %[[BLOCK_SIZES5]] to i8*
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // CHECK-LIFETIMES-NEXT: getelementptr inbounds [1 x i64], [1 x i64]* %[[BLOCK_SIZES5]], i64 0, i64 0
  // CHECK-LIFETIMES-LABEL: call spir_func i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], [1 x i32]* %[[BLOCK_SIZES5]], i32 0, i32 0
  // B32: store i32 %{{.*}}, i32* %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], [1 x i64]* %[[BLOCK_SIZES5]], i32 0, i32 0
  // B64: store i64 %{{.*}}, i64* %[[TMP]], align 8
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_varargs
  // COMMON-SAME: (%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %struct.ndrange_t* [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK5:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG5]] to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1,
  // B32-SAME: i32* %[[TMP]])
  // B64-SAME: i64* %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 l);

  // Emits global block literal [[BLG6]] and block kernel [[INVGK6]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK-LIFETIMES: [[LIFETIME_PTR:%[0-9]+]] = bitcast [3 x i64]* %[[BLOCK_SIZES6]] to i8*
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull [[LIFETIME_PTR]])
  // CHECK-LIFETIMES-NEXT: getelementptr inbounds [3 x i64], [3 x i64]* %[[BLOCK_SIZES6]], i64 0, i64 0
  // CHECK-LIFETIMES-LABEL: call spir_func i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull [[LIFETIME_PTR]])
  // B32: %[[TMP:.*]] = getelementptr [3 x i32], [3 x i32]* %[[BLOCK_SIZES6]], i32 0, i32 0
  // B32: store i32 1, i32* %[[TMP]], align 4
  // B32: %[[BLOCK_SIZES62:.*]] = getelementptr [3 x i32], [3 x i32]* %[[BLOCK_SIZES6]], i32 0, i32 1
  // B32: store i32 2, i32* %[[BLOCK_SIZES62]], align 4
  // B32: %[[BLOCK_SIZES63:.*]] = getelementptr [3 x i32], [3 x i32]* %[[BLOCK_SIZES6]], i32 0, i32 2
  // B32: store i32 4, i32* %[[BLOCK_SIZES63]], align 4
  // B64: %[[TMP:.*]] = getelementptr [3 x i64], [3 x i64]* %[[BLOCK_SIZES6]], i32 0, i32 0
  // B64: store i64 1, i64* %[[TMP]], align 8
  // B64: %[[BLOCK_SIZES62:.*]] = getelementptr [3 x i64], [3 x i64]* %[[BLOCK_SIZES6]], i32 0, i32 1
  // B64: store i64 2, i64* %[[BLOCK_SIZES62]], align 8
  // B64: %[[BLOCK_SIZES63:.*]] = getelementptr [3 x i64], [3 x i64]* %[[BLOCK_SIZES6]], i32 0, i32 2
  // B64: store i64 4, i64* %[[BLOCK_SIZES63]], align 8
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_varargs
  // COMMON-SAME: (%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %struct.ndrange_t* [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK6:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG6]] to i8 addrspace(1)*) to i8 addrspace(4)*), i32 3,
  // B32-SAME: i32* %[[TMP]])
  // B64-SAME: i64* %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p1, local void *p2, local void *p3) {
                   return;
                 },
                 1, 2, 4);

  // Emits global block literal [[BLG7]] and block kernel [[INVGK7]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t*, %opencl.queue_t** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK-LIFETIMES: [[LIFETIME_PTR:%[0-9]+]] = bitcast [1 x i64]* %[[BLOCK_SIZES7]] to i8*
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // CHECK-LIFETIMES-NEXT: getelementptr inbounds [1 x i64], [1 x i64]* %[[BLOCK_SIZES7]], i64 0, i64 0
  // CHECK-LIFETIMES-LABEL: call spir_func i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull [[LIFETIME_PTR]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], [1 x i32]* %[[BLOCK_SIZES7]], i32 0, i32 0
  // B32: store i32 0, i32* %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], [1 x i64]* %[[BLOCK_SIZES7]], i32 0, i32 0
  // B64: store i64 4294967296, i64* %[[TMP]], align 8
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_varargs
  // COMMON-SAME: (%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %struct.ndrange_t* [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK7:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG7]] to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1,
  // B32-SAME: i32* %[[TMP]])
  // B64-SAME: i64* %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 4294967296L);

  // Emits global block literal [[BLG8]] and invoke function [[INVG8]].
  // The full type of these expressions are long (and repeated elsewhere), so we
  // capture it as part of the regex for convenience and clarity.
  // COMMON: store %struct.__opencl_block_literal_generic addrspace(4)* addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG8]] to %struct.__opencl_block_literal_generic addrspace(1)*) to %struct.__opencl_block_literal_generic addrspace(4)*), %struct.__opencl_block_literal_generic addrspace(4)** %block_A
  void (^const block_A)(void) = ^{
    return;
  };

  // Emits global block literal [[BLG9]] and invoke function [[INVG9]].
  // COMMON: store %struct.__opencl_block_literal_generic addrspace(4)* addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG9]] to %struct.__opencl_block_literal_generic addrspace(1)*) to %struct.__opencl_block_literal_generic addrspace(4)*), %struct.__opencl_block_literal_generic addrspace(4)** %block_B
  void (^const block_B)(local void *) = ^(local void *a) {
    return;
  };

  // Uses global block literal [[BLG8]] and invoke function [[INVG8]].
  // COMMON: call spir_func void @__device_side_enqueue_block_invoke_11(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG8]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  block_A();

  // Emits global block literal [[BLG8]] and block kernel [[INVGK8]]. [[INVGK8]] calls [[INVG8]].
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_basic(
  // COMMON-SAME: %opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %struct.ndrange_t* byval(%struct.ndrange_t) [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK8:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG8]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  enqueue_kernel(default_queue, flags, ndrange, block_A);

  // Uses block kernel [[INVGK8]] and global block literal [[BLG8]].
  // COMMON: call spir_func i32 @__get_kernel_work_group_size_impl(
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK8]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG8]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  unsigned size = get_kernel_work_group_size(block_A);

  // Uses global block literal [[BLG8]] and invoke function [[INVG8]]. Make sure no redundant block literal and invoke functions are emitted.
  // COMMON: call spir_func void @__device_side_enqueue_block_invoke_11(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG8]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  block_A();

  // Make sure that block invoke function is resolved correctly after sequence of assignements.
  // COMMON: store %struct.__opencl_block_literal_generic addrspace(4)*
  // COMMON-SAME: addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)*
  // COMMON-SAME: bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BL_GLOBAL]] to %struct.__opencl_block_literal_generic addrspace(1)*)
  // COMMON-SAME: to %struct.__opencl_block_literal_generic addrspace(4)*),
  // COMMON-SAME: %struct.__opencl_block_literal_generic addrspace(4)** %b1,
  bl_t b1 = block_G;
  // COMMON: store %struct.__opencl_block_literal_generic addrspace(4)*
  // COMMON-SAME: addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)*
  // COMMON-SAME: bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BL_GLOBAL]] to %struct.__opencl_block_literal_generic addrspace(1)*)
  // COMMON-SAME: to %struct.__opencl_block_literal_generic addrspace(4)*),
  // COMMON-SAME: %struct.__opencl_block_literal_generic addrspace(4)** %b2,
  bl_t b2 = b1;
  // COMMON: call spir_func void @block_G_block_invoke(i8 addrspace(4)* addrspacecast (i8 addrspace(1)*
  // COMMON-SAME: bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BL_GLOBAL]] to i8 addrspace(1)*)
  // COOMON-SAME: to i8 addrspace(4)*), i8 addrspace(3)* null)
  b2(0);
  // Uses global block literal [[BL_GLOBAL]] and block kernel [[INV_G_K]]. [[INV_G_K]] calls [[INV_G]].
  // COMMON: call spir_func i32 @__get_kernel_preferred_work_group_size_multiple_impl(
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INV_G_K:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BL_GLOBAL]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_preferred_work_group_size_multiple(b2);

  void (^block_C)(void) = ^{
    callee(i, a);
  };
  // Emits block literal on stack and block kernel [[INVLK3]].
  // COMMON: store i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* [[INVL3:@__device_side_enqueue_block_invoke[^ ]*]] to i8*) to i8 addrspace(4)*), i8 addrspace(4)** %block.invoke
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[BL_I8:%[0-9]+]] ={{.*}} addrspacecast %struct.__opencl_block_literal_generic* {{.*}} to i8 addrspace(4)*
  // COMMON-LABEL: call spir_func i32 @__enqueue_kernel_basic(
  // COMMON-SAME: %opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %struct.ndrange_t* byval(%struct.ndrange_t) [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVLK3:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange, block_C);

  // Emits global block literal [[BLG9]] and block kernel [[INVGK9]]. [[INVGK9]] calls [[INV9]].
  // COMMON: call spir_func i32 @__get_kernel_work_group_size_impl(
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK9:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG9]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_work_group_size(block_B);

  // Uses global block literal [[BLG8]] and block kernel [[INVGK8]]. Make sure no redundant block literal ind invoke functions are emitted.
  // COMMON: call spir_func i32 @__get_kernel_preferred_work_group_size_multiple_impl(
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK8]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG8]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_preferred_work_group_size_multiple(block_A);

  // Uses global block literal [[BL_GLOBAL]] and block kernel [[INV_G_K]]. [[INV_G_K]] calls [[INV_G]].
  // COMMON: call spir_func i32 @__get_kernel_preferred_work_group_size_multiple_impl(
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INV_G_K:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BL_GLOBAL]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_preferred_work_group_size_multiple(block_G);

  // Emits global block literal [[BLG10]] and block kernel [[INVGK10]].
  // COMMON: call spir_func i32 @__get_kernel_max_sub_group_size_for_ndrange_impl(%struct.ndrange_t* {{[^,]+}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK10:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG10]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_max_sub_group_size_for_ndrange(ndrange, ^(){});

  // Emits global block literal [[BLG11]] and block kernel [[INVGK11]].
  // COMMON: call spir_func i32 @__get_kernel_sub_group_count_for_ndrange_impl(%struct.ndrange_t* {{[^,]+}},
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8* bitcast ({{.*}} [[INVGK11:[^ ]+_kernel]] to i8*) to i8 addrspace(4)*),
  // COMMON-SAME: i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* [[BLG11]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_sub_group_count_for_ndrange(ndrange, ^(){});
}

// COMMON: define internal spir_kernel void [[INVLK1]](i8 addrspace(4)* %0) #{{[0-9]+}} {
// COMMON: entry:
// COMMON:  call spir_func void @__device_side_enqueue_block_invoke(i8 addrspace(4)* %0)
// COMMON:  ret void
// COMMON: }
// COMMON: define internal spir_kernel void [[INVLK2]](i8 addrspace(4)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK1]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK2]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK3]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK4]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK5]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK6]](i8 addrspace(4)* %0, i8 addrspace(3)* %1, i8 addrspace(3)* %2, i8 addrspace(3)* %3) #{{[0-9]+}} {
// COMMON: entry:
// COMMON:  call spir_func void @__device_side_enqueue_block_invoke_9(i8 addrspace(4)* %0, i8 addrspace(3)* %1, i8 addrspace(3)* %2, i8 addrspace(3)* %3)
// COMMON:  ret void
// COMMON: }
// COMMON: define internal spir_kernel void [[INVGK7]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)*{{.*}})
// COMMON: define internal spir_func void [[INVG8]](i8 addrspace(4)*{{.*}})
// COMMON: define internal spir_func void [[INVG9]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)* %{{.*}})
// COMMON: define internal spir_kernel void [[INVGK8]](i8 addrspace(4)*{{.*}})
// COMMON: define internal spir_kernel void [[INV_G_K]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)*{{.*}})
// COMMON: define internal spir_kernel void [[INVLK3]](i8 addrspace(4)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK9]](i8 addrspace(4)*{{.*}}, i8 addrspace(3)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK10]](i8 addrspace(4)*{{.*}})
// COMMON: define internal spir_kernel void [[INVGK11]](i8 addrspace(4)*{{.*}})
