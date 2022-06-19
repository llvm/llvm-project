// RUN: %clang_cc1 -no-opaque-pointers -O0 -cl-std=CL2.0  -emit-llvm %s -o - | FileCheck %s
extern queue_t get_default_queue(void);

bool compare(void) {
  return 0 == get_default_queue() &&
         get_default_queue() == 0;
  // CHECK: icmp eq %opencl.queue_t* null, %{{.*}}
  // CHECK: icmp eq %opencl.queue_t* %{{.*}}, null
}

void func(queue_t q);

void init(void) {
  queue_t q = 0;
  func(0);
  // CHECK: store %opencl.queue_t* null, %opencl.queue_t** %q
  // CHECK: call void @func(%opencl.queue_t* null)
}
