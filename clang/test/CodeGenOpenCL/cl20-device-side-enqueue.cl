// RUN: %clang_cc1 %s -cl-std=CL2.0 -ffake-address-space-map -O0 -emit-llvm -o - | FileCheck %s

typedef void (^bl_t)(local void *);

const bl_t block_G = (bl_t) ^ (local void *a) {};

kernel void device_side_enqueue(global int *a, global int *b, int i) {
  // CHECK: %default_queue = alloca %opencl.queue_t*
  queue_t default_queue;
  // CHECK: %flags = alloca i32
  unsigned flags = 0;
  // CHECK: %ndrange = alloca %opencl.ndrange_t*
  ndrange_t ndrange;
  // CHECK: %clk_event = alloca %opencl.clk_event_t*
  clk_event_t clk_event;
  // CHECK: %event_wait_list = alloca %opencl.clk_event_t*
  clk_event_t event_wait_list;
  // CHECK: %event_wait_list2 = alloca [1 x %opencl.clk_event_t*]
  clk_event_t event_wait_list2[] = {clk_event};

  // CHECK: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t*, %opencl.queue_t** %default_queue
  // CHECK: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // CHECK: [[BL:%[0-9]+]] = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32{{.*}}, i32{{.*}}, i32{{.*}} }>* %block to void ()*
  // CHECK: [[BL_I8:%[0-9]+]] = bitcast void ()* [[BL]] to i8*
  // CHECK: call i32 @__enqueue_kernel_basic(%opencl.queue_t* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8* [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(void) {
                   a[i] = b[i];
                 });

  // CHECK: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t*, %opencl.queue_t** %default_queue
  // CHECK: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // CHECK: [[BL:%[0-9]+]] = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32{{.*}}, i32{{.*}}, i32{{.*}} }>* %block3 to void ()*
  // CHECK: [[BL_I8:%[0-9]+]] = bitcast void ()* [[BL]] to i8*
  // CHECK: call i32 @__enqueue_kernel_basic_events(%opencl.queue_t* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i32 2, %opencl.clk_event_t** %event_wait_list, %opencl.clk_event_t** %clk_event, i8* [[BL_I8]])
  enqueue_kernel(default_queue, flags, ndrange, 2, &event_wait_list, &clk_event,
                 ^(void) {
                   a[i] = b[i];
                 });

  // CHECK: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t*, %opencl.queue_t** %default_queue
  // CHECK: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // CHECK: call i32 (%opencl.queue_t*, i32, %opencl.ndrange_t*, i8*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }* @__block_literal_global{{(.[0-9]+)?}} to i8*), i32 1, i32 256)
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 256);
  char c;
  // CHECK: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t*, %opencl.queue_t** %default_queue
  // CHECK: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // CHECK: [[SIZE:%[0-9]+]] = zext i8 {{%[0-9]+}} to i32
  // CHECK: call i32 (%opencl.queue_t*, i32, %opencl.ndrange_t*, i8*, i32, ...) @__enqueue_kernel_vaargs(%opencl.queue_t* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i8* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }* @__block_literal_global{{(.[0-9]+)?}} to i8*), i32 1, i32 [[SIZE]])
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *p) {
                   return;
                 },
                 c);

  // CHECK: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t*, %opencl.queue_t** %default_queue
  // CHECK: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // CHECK: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x %opencl.clk_event_t*], [1 x %opencl.clk_event_t*]* %event_wait_list2, i32 0, i32 0
  // CHECK: call i32 (%opencl.queue_t*, i32, %opencl.ndrange_t*, i32, %opencl.clk_event_t**, %opencl.clk_event_t**, i8*, i32, ...) @__enqueue_kernel_events_vaargs(%opencl.queue_t* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i32 2, %opencl.clk_event_t** [[AD]], %opencl.clk_event_t** %clk_event, i8* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }* @__block_literal_global{{(.[0-9]+)?}} to i8*), i32 1, i32 256)
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 256);

  // CHECK: [[DEF_Q:%[0-9]+]] = load %opencl.queue_t*, %opencl.queue_t** %default_queue
  // CHECK: [[FLAGS:%[0-9]+]] = load i32, i32* %flags
  // CHECK: [[NDR:%[0-9]+]] = load %opencl.ndrange_t*, %opencl.ndrange_t** %ndrange
  // CHECK: [[AD:%arraydecay[0-9]*]] = getelementptr inbounds [1 x %opencl.clk_event_t*], [1 x %opencl.clk_event_t*]* %event_wait_list2, i32 0, i32 0
  // CHECK: [[SIZE:%[0-9]+]] = zext i8 {{%[0-9]+}} to i32
  // CHECK: call i32 (%opencl.queue_t*, i32, %opencl.ndrange_t*, i32, %opencl.clk_event_t**, %opencl.clk_event_t**, i8*, i32, ...) @__enqueue_kernel_events_vaargs(%opencl.queue_t* [[DEF_Q]], i32 [[FLAGS]], %opencl.ndrange_t* [[NDR]], i32 2, %opencl.clk_event_t** [[AD]], %opencl.clk_event_t** %clk_event, i8* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }* @__block_literal_global{{(.[0-9]+)?}} to i8*), i32 1, i32 [[SIZE]])
  enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
                 ^(local void *p) {
                   return;
                 },
                 c);

  void (^const block_A)(void) = ^{
    return;
  };
  void (^const block_B)(local void *) = ^(local void *a) {
    return;
  };

  // CHECK: [[BL:%[0-9]+]] = load void ()*, void ()** %block_A
  // CHECK: [[BL_I8:%[0-9]+]] = bitcast void ()* [[BL]] to i8*
  // CHECK: call i32 @__get_kernel_work_group_size_impl(i8* [[BL_I8]])
  unsigned size = get_kernel_work_group_size(block_A);
  // CHECK: [[BL:%[0-9]+]] = load void (i8 addrspace(2)*)*, void (i8 addrspace(2)*)** %block_B
  // CHECK: [[BL_I8:%[0-9]+]] = bitcast void (i8 addrspace(2)*)* [[BL]] to i8*
  // CHECK: call i32 @__get_kernel_work_group_size_impl(i8* [[BL_I8]])
  size = get_kernel_work_group_size(block_B);
  // CHECK: [[BL:%[0-9]+]] = load void ()*, void ()** %block_A
  // CHECK: [[BL_I8:%[0-9]+]] = bitcast void ()* [[BL]] to i8*
  // CHECK: call i32 @__get_kernel_preferred_work_group_multiple_impl(i8* [[BL_I8]])
  size = get_kernel_preferred_work_group_size_multiple(block_A);
  // CHECK: [[BL:%[0-9]+]] = load void (i8 addrspace(2)*)*, void (i8 addrspace(2)*)* addrspace(1)* @block_G
  // CHECK: [[BL_I8:%[0-9]+]] = bitcast void (i8 addrspace(2)*)* [[BL]] to i8*
  // CHECK: call i32 @__get_kernel_preferred_work_group_multiple_impl(i8* [[BL_I8]])
  size = get_kernel_preferred_work_group_size_multiple(block_G);
}
