// RUN: %clang_cc1 %s -cl-std=CL2.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "spir-unknown-unknown" | FileCheck %s --check-prefix=COMMON --check-prefix=B32
// RUN: %clang_cc1 %s -cl-std=CL2.0 -ffake-address-space-map -O0 -emit-llvm -o - -triple "spir64-unknown-unknown" | FileCheck %s --check-prefix=COMMON --check-prefix=B64

typedef void (^bl_t)(local void *);
// N.B. The check here only exists to set BL_GLOBAL
// COMMON: @block_G =  addrspace(1) constant void (i8 addrspace(3)*) addrspace(4)* addrspacecast (void (i8 addrspace(3)*) addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* [[BL_GLOBAL:@__block_literal_global(\.[0-9]+)?]] to void (i8 addrspace(3)*) addrspace(1)*) to void (i8 addrspace(3)*) addrspace(4)*)
const bl_t block_G = (bl_t) ^ (local void *a) {};

kernel void device_side_enqueue(global int *a, global int *b, int i) {
  // COMMON: %default_queue = alloca %opencl.queue_t*
  queue_t default_queue;
  // COMMON: %flags = alloca i32
  unsigned flags = 0;
  // COMMON: %ndrange = alloca %opencl.ndrange_t*
  ndrange_t ndrange;
  // COMMON: %clk_event = alloca %opencl.clk_event_t*
  clk_event_t clk_event;
  // COMMON: %event_wait_list = alloca %opencl.clk_event_t*
  clk_event_t event_wait_list;
  // COMMON: %event_wait_list2 = alloca [1 x %opencl.clk_event_t*]
  clk_event_t event_wait_list2[] = {clk_event};

  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // COMMON: [[BL:%[0-9]+]] = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor addrspace(2)*, i32{{.*}}, i32{{.*}}, i32{{.*}} }>* %block to void ()*
  // COMMON: [[BL_I8:%[0-9]+]] = addrspacecast void ()* [[BL]] to i8 addrspace(4)*
  // COMMON: call i32 @__enqueue_kernel_basic(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(void) {
                   a[i] = b[i];
                 });

  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // COMMON: [[WAIT_EVNT:%[0-9]+]] = addrspacecast %opencl.clk_event_t{{.*}}** %event_wait_list to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // COMMON: [[EVNT:%[0-9]+]] = addrspacecast %opencl.clk_event_t{{.*}}** %clk_event to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // COMMON: [[BL:%[0-9]+]] = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor addrspace(2)*, i32{{.*}}, i32{{.*}}, i32{{.*}} }>* %block3 to void ()*
  // COMMON: [[BL_I8:%[0-9]+]] = addrspacecast void ()* [[BL]] to i8 addrspace(4)*
  // COMMON: call i32 @__enqueue_kernel_basic_events(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]],  %opencl.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t{{.*}}* addrspace(4)* [[WAIT_EVNT]], %opencl.clk_event_t{{.*}}* addrspace(4)* [[EVNT]], i8 addrspace(4)* [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange, 2, &event_wait_list, &clk_event,
                 ^(void) {
                   a[i] = b[i];
                 });

  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // B32: call i32 (%opencl.queue_t{{.*}}*, i32, %opencl.ndrange_t*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i32 256)
  // B64: call i32 (%opencl.queue_t{{.*}}*, i32, %opencl.ndrange_t*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i64 256)
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 256);
  char c;
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // B32: [[SIZE:%[0-9]+]] = zext i8 {{%[0-9]+}} to i32
  // B32: call i32 (%opencl.queue_t{{.*}}*, i32, %opencl.ndrange_t*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i32 [[SIZE]])
  // B64: [[SIZE:%[0-9]+]] = zext i8 {{%[0-9]+}} to i64
  // B64: call i32 (%opencl.queue_t{{.*}}*, i32, %opencl.ndrange_t*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i64 [[SIZE]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 c);

  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // COMMON: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x %opencl.clk_event_t*], [1 x %opencl.clk_event_t*]* %event_wait_list2, i32 0, i32 0
  // COMMON: [[WAIT_EVNT:%[0-9]+]] = addrspacecast %opencl.clk_event_t{{.*}}** [[AD]] to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // COMMON: [[EVNT:%[0-9]+]]  = addrspacecast %opencl.clk_event_t{{.*}}** %clk_event to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // B32: call i32 (%opencl.queue_t{{.*}}*, i32,  %opencl.ndrange_t*, i32, %opencl.clk_event_t{{.*}}* addrspace(4)*, %opencl.clk_event_t{{.*}}* addrspace(4)*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_events_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]],  %opencl.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t{{.*}} [[WAIT_EVNT]], %opencl.clk_event_t{{.*}} [[EVNT]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i32 256)
  // B64: call i32 (%opencl.queue_t{{.*}}*, i32,  %opencl.ndrange_t*, i32, %opencl.clk_event_t{{.*}}* addrspace(4)*, %opencl.clk_event_t{{.*}}* addrspace(4)*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_events_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]],  %opencl.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t{{.*}} [[WAIT_EVNT]], %opencl.clk_event_t{{.*}} [[EVNT]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i64 256)
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 256);

  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // COMMON: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x %opencl.clk_event_t*], [1 x %opencl.clk_event_t*]* %event_wait_list2, i32 0, i32 0
  // COMMON: [[WAIT_EVNT:%[0-9]+]] = addrspacecast %opencl.clk_event_t{{.*}}** [[AD]] to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // COMMON: [[EVNT:%[0-9]+]]  = addrspacecast %opencl.clk_event_t{{.*}}** %clk_event to %opencl.clk_event_t{{.*}}* addrspace(4)*
  // B32: [[SIZE:%[0-9]+]] = zext i8 {{%[0-9]+}} to i32
  // B32: call i32 (%opencl.queue_t{{.*}}*, i32,  %opencl.ndrange_t*, i32, %opencl.clk_event_t{{.*}}* addrspace(4)*, %opencl.clk_event_t{{.*}}* addrspace(4)*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_events_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]],  %opencl.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t{{.*}}* addrspace(4)* [[WAIT_EVNT]], %opencl.clk_event_t{{.*}}* addrspace(4)* [[EVNT]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i32 [[SIZE]])
  // B64: [[SIZE:%[0-9]+]] = zext i8 {{%[0-9]+}} to i64
  // B64: call i32 (%opencl.queue_t{{.*}}*, i32,  %opencl.ndrange_t*, i32, %opencl.clk_event_t{{.*}}* addrspace(4)*, %opencl.clk_event_t{{.*}}* addrspace(4)*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_events_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]],  %opencl.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t{{.*}}* addrspace(4)* [[WAIT_EVNT]], %opencl.clk_event_t{{.*}}* addrspace(4)* [[EVNT]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i64 [[SIZE]])
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 c);

  long l;
  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t{{.*}}*, %opencl.queue_t{{.*}}** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // B32: [[SIZE:%[0-9]+]] = trunc i64 {{%[0-9]+}} to i32
  // B32: call i32 (%opencl.queue_t{{.*}}*, i32, %opencl.ndrange_t*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i32 [[SIZE]])
  // B64: [[SIZE:%[0-9]+]] = load i64, i64* %l
  // B64: call i32 (%opencl.queue_t{{.*}}*, i32, %opencl.ndrange_t*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i64 [[SIZE]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 l);

  // COMMON: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t*, %opencl.queue_t** %default_queue
  // COMMON: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // COMMON: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // B32: call i32 (%opencl.queue_t{{.*}}*, i32, %opencl.ndrange_t*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i32 0)
  // B64: call i32 (%opencl.queue_t{{.*}}*, i32, %opencl.ndrange_t*, i8 addrspace(4)*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t{{.*}}* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* @__block_literal_global{{(.[0-9]+)?}} to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i64 4294967296)
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 4294967296L);

  // The full type of these expressions are long (and repeated elsewhere), so we
  // capture it as part of the regex for convenience and clarity.
  // COMMON: store void () addrspace(4)* addrspacecast (void () addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* [[BL_A:@__block_literal_global(\.[0-9]+)?]] to void () addrspace(1)*) to void () addrspace(4)*), void () addrspace(4)** %block_A
  void (^const block_A)(void) = ^{
    return;
  };

  // COMMON: store void (i8 addrspace(3)*) addrspace(4)* addrspacecast (void (i8 addrspace(3)*) addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* [[BL_B:@__block_literal_global(\.[0-9]+)?]] to void (i8 addrspace(3)*) addrspace(1)*) to void (i8 addrspace(3)*) addrspace(4)*), void (i8 addrspace(3)*) addrspace(4)** %block_B
  void (^const block_B)(local void *) = ^(local void *a) {
    return;
  };

  // COMMON: call i32 @__get_kernel_work_group_size_impl(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* [[BL_A]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  unsigned size = get_kernel_work_group_size(block_A);
  // COMMON: call i32 @__get_kernel_work_group_size_impl(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* [[BL_B]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_work_group_size(block_B);
  // COMMON: call i32 @__get_kernel_preferred_work_group_multiple_impl(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* [[BL_A]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_preferred_work_group_size_multiple(block_A);
  // COMMON: call i32 @__get_kernel_preferred_work_group_multiple_impl(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(2)* } addrspace(1)* [[BL_GLOBAL]] to i8 addrspace(1)*) to i8 addrspace(4)*))
  size = get_kernel_preferred_work_group_size_multiple(block_G);
}
