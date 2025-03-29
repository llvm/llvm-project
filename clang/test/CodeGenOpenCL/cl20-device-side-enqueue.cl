// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "spir-unknown-unknown" | FileCheck %s --check-prefixes=COMMON,B32,SPIR
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "spir64-unknown-unknown" | FileCheck %s --check-prefixes=COMMON,B64,SPIR
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -ffake-address-space-map -O1 -emit-llvm -o - -triple "spir64-unknown-unknown" | FileCheck %s --check-prefix=CHECK-LIFETIMES
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL3.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "spir-unknown-unknown" | FileCheck %s --check-prefixes=COMMON,B32,SPIR
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL3.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "spir64-unknown-unknown" | FileCheck %s --check-prefixes=COMMON,B64,SPIR
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL3.0 -ffake-address-space-map -O1 -emit-llvm -o - -triple "spir64-unknown-unknown" | FileCheck %s --check-prefix=CHECK-LIFETIMES
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "x86_64-unknown-linux-gnu" | FileCheck %s --check-prefixes=COMMON,B64,X86
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL3.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "x86_64-unknown-linux-gnu" | FileCheck %s --check-prefixes=COMMON,B64,X86
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL3.0 -ffake-address-space-map -O1 -emit-llvm -o - -triple "x86_64-unknown-linux-gnu" | FileCheck %s --check-prefix=CHECK-LIFETIMES

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

typedef void (^bl_t)(local void *);
typedef struct {int a;} ndrange_t;

// For a block global variable, first emit the block literal as a global variable, then emit the block variable itself.
// COMMON: [[BL_GLOBAL:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr [[INV_G:@[^ ]+]] to ptr addrspace(4)) }
// COMMON: @block_G ={{.*}} addrspace(1) constant ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BL_GLOBAL]] to ptr addrspace(4))

// For anonymous blocks without captures, emit block literals as global variable.
// COMMON: [[BLG0:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG1:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG2:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG3:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG4:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG5:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG6:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG7:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG8:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr [[INVG8:@[^ ]+]] to ptr addrspace(4)) }
// COMMON: [[BLG9:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr [[INVG9:@[^ ]+]] to ptr addrspace(4)) }
// COMMON: [[BLG10:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }
// COMMON: [[BLG11:@__block_literal_global[^ ]*]] = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr addrspace(4) addrspacecast (ptr {{@[^ ]+}} to ptr addrspace(4)) }

// Emits block literal [[BL_GLOBAL]], invoke function [[INV_G]] and global block variable @block_G
// COMMON: define internal {{(spir_func )?}}void [[INV_G]](ptr addrspace(4) %{{.*}}, ptr addrspace(3) %{{.*}})
const bl_t block_G = (bl_t) ^ (local void *a) {};

void callee(int id, __global int *out) {
  out[id] = id;
}

// COMMON-LABEL: define{{.*}} spir_kernel void @device_side_enqueue(ptr addrspace(1) align 4 %{{.*}}, ptr addrspace(1) align 4 %b, i32 %i)
kernel void device_side_enqueue(global int *a, global int *b, int i) {
  // SPIR: %default_queue = alloca target("spirv.Queue")
  // X86: %default_queue = alloca ptr
  queue_t default_queue;
  // COMMON: %flags = alloca i32
  unsigned flags = 0;
  // COMMON: %ndrange = alloca %struct.ndrange_t
  ndrange_t ndrange;
  // SPIR: %clk_event = alloca target("spirv.DeviceEvent")
  // X86: %clk_event = alloca ptr
  clk_event_t clk_event;
  // SPIR: %event_wait_list = alloca target("spirv.DeviceEvent")
  // X86: %event_wait_list = alloca ptr
  clk_event_t event_wait_list;
  // SPIR: %event_wait_list2 = alloca [1 x target("spirv.DeviceEvent")]
  // X86: %event_wait_list2 = alloca [1 x ptr]
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
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // COMMON: store ptr addrspace(4) addrspacecast (ptr [[INVL1:@__device_side_enqueue_block_invoke[^ ]*]] to ptr addrspace(4)), ptr %block.invoke
  // COMMON: [[BL_I8:%[0-9]+]] ={{.*}} addrspacecast ptr %block to ptr addrspace(4)
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_basic(
  // SPIR-SAME: target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // X86-SAME: ptr [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVLK1:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(void) {
                   a[i] = b[i];
                 });

  // Emits block literal on stack and block kernel [[INVLK2]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // COMMON: [[WAIT_EVNT:%[0-9]+]] ={{.*}} addrspacecast ptr %event_wait_list to ptr addrspace(4)
  // COMMON: [[EVNT:%[0-9]+]] ={{.*}} addrspacecast ptr %clk_event to ptr addrspace(4)
  // COMMON: store ptr addrspace(4) addrspacecast (ptr [[INVL2:@__device_side_enqueue_block_invoke[^ ]*]] to ptr addrspace(4)), ptr %block.invoke
  // COMMON: [[BL_I8:%[0-9]+]] ={{.*}} addrspacecast ptr %block4 to ptr addrspace(4)
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_basic_events
  // SPIR-SAME: (target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]],  ptr {{.*}}, i32 2, ptr addrspace(4) [[WAIT_EVNT]], ptr addrspace(4) [[EVNT]],
  // X86-SAME: (ptr [[DEF_Q]], i32 [[FLAGS]],  ptr {{.*}}, i32 2, ptr addrspace(4) [[WAIT_EVNT]], ptr addrspace(4) [[EVNT]],
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVLK2:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange, 2, &event_wait_list, &clk_event,
                 ^(void) {
                   a[i] = b[i];
                 });

  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_basic_events
  // SPIR-SAME: (target("spirv.Queue") {{%[0-9]+}}, i32 {{%[0-9]+}}, ptr {{.*}}, i32 1, ptr addrspace(4) null, ptr addrspace(4) null,
  // X86-SAME: (ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, ptr {{.*}}, i32 1, ptr addrspace(4) null, ptr addrspace(4) null,
  enqueue_kernel(default_queue, flags, ndrange, 1, 0, 0,
                 ^(void) {
                   return;
                 });

  // Emits global block literal [[BLG1]] and block kernel [[INVGK1]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // CHECK-LIFETIMES: call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %[[BLOCK_SIZES1]])
  // CHECK-LIFETIMES-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %[[BLOCK_SIZES1]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], ptr %[[BLOCK_SIZES1]], i32 0, i32 0
  // B32: store i32 256, ptr %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], ptr %[[BLOCK_SIZES1]], i32 0, i32 0
  // B64: store i64 256, ptr %[[TMP]], align 8
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs(
  // SPIR-SAME: target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // X86-SAME: ptr [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK1:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG1]] to ptr addrspace(4)), i32 1,
  // B32-SAME: ptr %[[TMP]])
  // B64-SAME: ptr %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 256);

  char c;
  // Emits global block literal [[BLG2]] and block kernel [[INVGK2]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %[[BLOCK_SIZES2]])
  // CHECK-LIFETIMES-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %[[BLOCK_SIZES2]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], ptr %[[BLOCK_SIZES2]], i32 0, i32 0
  // B32: store i32 %{{.*}}, ptr %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], ptr %[[BLOCK_SIZES2]], i32 0, i32 0
  // B64: store i64 %{{.*}}, ptr %[[TMP]], align 8
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs(
  // SPIR-SAME: target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // X86-SAME: ptr [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK2:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG2]] to ptr addrspace(4)), i32 1,
  // B32-SAME: ptr %[[TMP]])
  // B64-SAME: ptr %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 c);

  // Emits global block literal [[BLG3]] and block kernel [[INVGK3]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // SPIR: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x target("spirv.DeviceEvent")], ptr %event_wait_list2, i{{32|64}} 0, i{{32|64}} 0
  // X86: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x ptr], ptr %event_wait_list2, i{{32|64}} 0, i{{32|64}} 0
  // COMMON: [[WAIT_EVNT:%[0-9]+]] ={{.*}} addrspacecast ptr [[AD]] to ptr addrspace(4)
  // COMMON: [[EVNT:%[0-9]+]]  ={{.*}} addrspacecast ptr %clk_event to ptr addrspace(4)
  // CHECK-LIFETIMES: call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %[[BLOCK_SIZES3]])
  // CHECK-LIFETIMES-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_events_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %[[BLOCK_SIZES3]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], ptr %[[BLOCK_SIZES3]], i32 0, i32 0
  // B32: store i32 256, ptr %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], ptr %[[BLOCK_SIZES3]], i32 0, i32 0
  // B64: store i64 256, ptr %[[TMP]], align 8
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_events_varargs
  // SPIR-SAME: (target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]],  ptr {{.*}}, i32 2, ptr addrspace(4) [[WAIT_EVNT]], ptr addrspace(4) [[EVNT]],
  // X86-SAME: (ptr [[DEF_Q]], i32 [[FLAGS]],  ptr {{.*}}, i32 2, ptr addrspace(4) [[WAIT_EVNT]], ptr addrspace(4) [[EVNT]],
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK3:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG3]] to ptr addrspace(4)), i32 1,
  // B32-SAME: ptr %[[TMP]])
  // B64-SAME: ptr %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 256);

  // Emits global block literal [[BLG4]] and block kernel [[INVGK4]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // SPIR: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x target("spirv.DeviceEvent")], ptr %event_wait_list2, i{{32|64}} 0, i{{32|64}} 0
  // X86: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x ptr], ptr %event_wait_list2, i{{32|64}} 0, i{{32|64}} 0
  // COMMON: [[WAIT_EVNT:%[0-9]+]] ={{.*}} addrspacecast ptr [[AD]] to ptr addrspace(4)
  // COMMON: [[EVNT:%[0-9]+]]  ={{.*}} addrspacecast ptr %clk_event to ptr addrspace(4)
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %[[BLOCK_SIZES4]])
  // CHECK-LIFETIMES-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_events_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %[[BLOCK_SIZES4]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], ptr %[[BLOCK_SIZES4]], i32 0, i32 0
  // B32: store i32 %{{.*}}, ptr %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], ptr %[[BLOCK_SIZES4]], i32 0, i32 0
  // B64: store i64 %{{.*}}, ptr %[[TMP]], align 8
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_events_varargs
  // SPIR-SAME: (target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]],  ptr {{.*}}, i32 2, ptr addrspace(4) [[WAIT_EVNT]], ptr addrspace(4) [[EVNT]],
  // X86-SAME: (ptr [[DEF_Q]], i32 [[FLAGS]],  ptr {{.*}}, i32 2, ptr addrspace(4) [[WAIT_EVNT]], ptr addrspace(4) [[EVNT]],
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK4:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG4]] to ptr addrspace(4)), i32 1,
  // B32-SAME: ptr %[[TMP]])
  // B64-SAME: ptr %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 c);

  long l;
  // Emits global block literal [[BLG5]] and block kernel [[INVGK5]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %[[BLOCK_SIZES5]])
  // CHECK-LIFETIMES-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %[[BLOCK_SIZES5]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], ptr %[[BLOCK_SIZES5]], i32 0, i32 0
  // B32: store i32 %{{.*}}, ptr %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], ptr %[[BLOCK_SIZES5]], i32 0, i32 0
  // B64: store i64 %{{.*}}, ptr %[[TMP]], align 8
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs
  // SPIR-SAME: (target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // X86-SAME: (ptr [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK5:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG5]] to ptr addrspace(4)), i32 1,
  // B32-SAME: ptr %[[TMP]])
  // B64-SAME: ptr %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 l);

  // Emits global block literal [[BLG6]] and block kernel [[INVGK6]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %[[BLOCK_SIZES6]])
  // CHECK-LIFETIMES-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %[[BLOCK_SIZES6]])
  // B32: %[[TMP:.*]] = getelementptr [3 x i32], ptr %[[BLOCK_SIZES6]], i32 0, i32 0
  // B32: store i32 1, ptr %[[TMP]], align 4
  // B32: %[[BLOCK_SIZES62:.*]] = getelementptr [3 x i32], ptr %[[BLOCK_SIZES6]], i32 0, i32 1
  // B32: store i32 2, ptr %[[BLOCK_SIZES62]], align 4
  // B32: %[[BLOCK_SIZES63:.*]] = getelementptr [3 x i32], ptr %[[BLOCK_SIZES6]], i32 0, i32 2
  // B32: store i32 4, ptr %[[BLOCK_SIZES63]], align 4
  // B64: %[[TMP:.*]] = getelementptr [3 x i64], ptr %[[BLOCK_SIZES6]], i32 0, i32 0
  // B64: store i64 1, ptr %[[TMP]], align 8
  // B64: %[[BLOCK_SIZES62:.*]] = getelementptr [3 x i64], ptr %[[BLOCK_SIZES6]], i32 0, i32 1
  // B64: store i64 2, ptr %[[BLOCK_SIZES62]], align 8
  // B64: %[[BLOCK_SIZES63:.*]] = getelementptr [3 x i64], ptr %[[BLOCK_SIZES6]], i32 0, i32 2
  // B64: store i64 4, ptr %[[BLOCK_SIZES63]], align 8
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs
  // SPIR-SAME: (target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // X86-SAME: (ptr [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK6:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG6]] to ptr addrspace(4)), i32 3,
  // B32-SAME: ptr %[[TMP]])
  // B64-SAME: ptr %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p1, local void *p2, local void *p3) {
                   return;
                 },
                 1, 2, 4);

  // Emits global block literal [[BLG7]] and block kernel [[INVGK7]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %[[BLOCK_SIZES7]])
  // CHECK-LIFETIMES-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs(
  // CHECK-LIFETIMES-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %[[BLOCK_SIZES7]])
  // B32: %[[TMP:.*]] = getelementptr [1 x i32], ptr %[[BLOCK_SIZES7]], i32 0, i32 0
  // B32: store i32 0, ptr %[[TMP]], align 4
  // B64: %[[TMP:.*]] = getelementptr [1 x i64], ptr %[[BLOCK_SIZES7]], i32 0, i32 0
  // B64: store i64 4294967296, ptr %[[TMP]], align 8
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_varargs
  // SPIR-SAME: (target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // X86-SAME: (ptr [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK7:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG7]] to ptr addrspace(4)), i32 1,
  // B32-SAME: ptr %[[TMP]])
  // B64-SAME: ptr %[[TMP]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 4294967296L);

  // Emits global block literal [[BLG8]] and invoke function [[INVG8]].
  // The full type of these expressions are long (and repeated elsewhere), so we
  // capture it as part of the regex for convenience and clarity.
  // COMMON: store ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG8]] to ptr addrspace(4)), ptr %block_A
  void (^const block_A)(void) = ^{
    return;
  };

  // Emits global block literal [[BLG9]] and invoke function [[INVG9]].
  // COMMON: store ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG9]] to ptr addrspace(4)), ptr %block_B
  void (^const block_B)(local void *) = ^(local void *a) {
    return;
  };

  // Uses global block literal [[BLG8]] and invoke function [[INVG8]].
  // COMMON: call {{(spir_func )?}}void @__device_side_enqueue_block_invoke_11(ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG8]] to ptr addrspace(4))) [[INVOKE_ATTR:#[0-9]+]]
  block_A();

  // Emits global block literal [[BLG8]] and block kernel [[INVGK8]]. [[INVGK8]] calls [[INVG8]].
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_basic(
  // SPIR-SAME: target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // X86-SAME: ptr [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK8:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG8]] to ptr addrspace(4)))
  enqueue_kernel(default_queue, flags, ndrange, block_A);

  // Uses block kernel [[INVGK8]] and global block literal [[BLG8]].
  // COMMON: call {{(spir_func )?}}i32 @__get_kernel_work_group_size_impl(
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK8]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG8]] to ptr addrspace(4)))
  unsigned size = get_kernel_work_group_size(block_A);

  // Uses global block literal [[BLG8]] and invoke function [[INVG8]]. Make sure no redundant block literal and invoke functions are emitted.
  // COMMON: call {{(spir_func )?}}void @__device_side_enqueue_block_invoke_11(ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG8]] to ptr addrspace(4)))
  block_A();

  // Make sure that block invoke function is resolved correctly after sequence of assignements.
  // COMMON: store ptr addrspace(4)
  // COMMON-SAME: addrspacecast (ptr addrspace(1)
  // COMMON-SAME: [[BL_GLOBAL]]
  // COMMON-SAME: to ptr addrspace(4)),
  // COMMON-SAME: ptr %b1,
  bl_t b1 = block_G;
  // COMMON: store ptr addrspace(4)
  // COMMON-SAME: addrspacecast (ptr addrspace(1)
  // COMMON-SAME: [[BL_GLOBAL]]
  // COMMON-SAME: to ptr addrspace(4)),
  // COMMON-SAME: ptr %b2,
  bl_t b2 = b1;
  // COMMON: call {{(spir_func )?}}void @block_G_block_invoke(ptr addrspace(4) addrspacecast (ptr addrspace(1)
  // COMMON-SAME: [[BL_GLOBAL]]
  // COOMON-SAME: to ptr addrspace(4)), ptr addrspace(3) null)
  b2(0);
  // Uses global block literal [[BL_GLOBAL]] and block kernel [[INV_G_K]]. [[INV_G_K]] calls [[INV_G]].
  // COMMON: call {{(spir_func )?}}i32 @__get_kernel_preferred_work_group_size_multiple_impl(
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INV_G_K:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BL_GLOBAL]] to ptr addrspace(4)))
  size = get_kernel_preferred_work_group_size_multiple(b2);

  void (^block_C)(void) = ^{
    callee(i, a);
  };
  // Emits block literal on stack and block kernel [[INVLK3]].
  // COMMON: store ptr addrspace(4) addrspacecast (ptr [[INVL3:@__device_side_enqueue_block_invoke[^ ]*]] to ptr addrspace(4)), ptr %block.invoke
  // SPIR: [[DEF_Q:%[0-9]+]] = load target("spirv.Queue"), ptr %default_queue
  // X86: [[DEF_Q:%[0-9]+]] = load ptr, ptr %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, ptr %flags
  // COMMON: [[BL_I8:%[0-9]+]] ={{.*}} addrspacecast ptr {{.*}} to ptr addrspace(4)
  // COMMON-LABEL: call {{(spir_func )?}}i32 @__enqueue_kernel_basic(
  // SPIR-SAME: target("spirv.Queue") [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // X86-SAME: ptr [[DEF_Q]], i32 [[FLAGS]], ptr [[NDR]]{{([0-9]+)?}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVLK3:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange, block_C);

  // Emits global block literal [[BLG9]] and block kernel [[INVGK9]]. [[INVGK9]] calls [[INV9]].
  // COMMON: call {{(spir_func )?}}i32 @__get_kernel_work_group_size_impl(
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK9:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG9]] to ptr addrspace(4)))
  size = get_kernel_work_group_size(block_B);

  // Uses global block literal [[BLG8]] and block kernel [[INVGK8]]. Make sure no redundant block literal ind invoke functions are emitted.
  // COMMON: call {{(spir_func )?}}i32 @__get_kernel_preferred_work_group_size_multiple_impl(
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK8]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG8]] to ptr addrspace(4)))
  size = get_kernel_preferred_work_group_size_multiple(block_A);

  // Uses global block literal [[BL_GLOBAL]] and block kernel [[INV_G_K]]. [[INV_G_K]] calls [[INV_G]].
  // COMMON: call {{(spir_func )?}}i32 @__get_kernel_preferred_work_group_size_multiple_impl(
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INV_G_K:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BL_GLOBAL]] to ptr addrspace(4)))
  size = get_kernel_preferred_work_group_size_multiple(block_G);

  // Emits global block literal [[BLG10]] and block kernel [[INVGK10]].
  // COMMON: call {{(spir_func )?}}i32 @__get_kernel_max_sub_group_size_for_ndrange_impl(ptr {{[^,]+}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK10:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG10]] to ptr addrspace(4)))
  size = get_kernel_max_sub_group_size_for_ndrange(ndrange, ^(){});

  // Emits global block literal [[BLG11]] and block kernel [[INVGK11]].
  // COMMON: call {{(spir_func )?}}i32 @__get_kernel_sub_group_count_for_ndrange_impl(ptr {{[^,]+}},
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr [[INVGK11:[^ ]+_kernel]] to ptr addrspace(4)),
  // COMMON-SAME: ptr addrspace(4) addrspacecast (ptr addrspace(1) [[BLG11]] to ptr addrspace(4)))
  size = get_kernel_sub_group_count_for_ndrange(ndrange, ^(){});
}

// COMMON: define spir_kernel void [[INVLK1]](ptr addrspace(4) %0) #{{[0-9]+}} {
// COMMON: entry:
// COMMON:  call {{(spir_func )?}}void @__device_side_enqueue_block_invoke(ptr addrspace(4) %0)
// COMMON:  ret void
// COMMON: }
// COMMON: define spir_kernel void [[INVLK2]](ptr addrspace(4){{.*}})
// COMMON: define spir_kernel void [[INVGK1]](ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})  [[INVOKE_KERNEL_ATTR:#[0-9]+]]
// COMMON: define spir_kernel void [[INVGK2]](ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})
// COMMON: define spir_kernel void [[INVGK3]](ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})
// COMMON: define spir_kernel void [[INVGK4]](ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})
// COMMON: define spir_kernel void [[INVGK5]](ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})
// COMMON: define spir_kernel void [[INVGK6]](ptr addrspace(4) %0, ptr addrspace(3) %1, ptr addrspace(3) %2, ptr addrspace(3) %3) #{{[0-9]+}} {
// COMMON: entry:
// COMMON:  call {{(spir_func )?}}void @__device_side_enqueue_block_invoke_9(ptr addrspace(4) %0, ptr addrspace(3) %1, ptr addrspace(3) %2, ptr addrspace(3) %3)
// COMMON:  ret void
// COMMON: }
// COMMON: define spir_kernel void [[INVGK7]](ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})
// COMMON: define internal {{(spir_func )?}}void [[INVG8]](ptr addrspace(4){{.*}}) [[INVG8_INVOKE_FUNC_ATTR:#[0-9]+]]
// COMMON: define internal {{(spir_func )?}}void [[INVG9]](ptr addrspace(4){{.*}}, ptr addrspace(3) %{{.*}})
// COMMON: define spir_kernel void [[INVGK8]](ptr addrspace(4){{.*}})
// COMMON: define spir_kernel void [[INV_G_K]](ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})
// COMMON: define spir_kernel void [[INVLK3]](ptr addrspace(4){{.*}})
// COMMON: define spir_kernel void [[INVGK9]](ptr addrspace(4){{.*}}, ptr addrspace(3){{.*}})
// COMMON: define spir_kernel void [[INVGK10]](ptr addrspace(4){{.*}})
// COMMON: define spir_kernel void [[INVGK11]](ptr addrspace(4){{.*}})

// SPIR: attributes [[INVG8_INVOKE_FUNC_ATTR]] = { convergent noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// SPIR: attributes [[INVOKE_KERNEL_ATTR]] = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// X86: attributes [[INVG8_INVOKE_FUNC_ATTR]] = { convergent noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="{{[^"]*}}" }
// X86: attributes [[INVOKE_KERNEL_ATTR]] = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="{{[^"]*}}" }
