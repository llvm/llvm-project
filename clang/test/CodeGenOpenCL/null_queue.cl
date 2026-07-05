// RUN: %clang_cc1 -O0 -cl-std=CL2.0  -emit-llvm %s -o - | FileCheck %s
extern queue_t get_default_queue(void);

bool compare(void) {
  return 0 == get_default_queue() &&
         get_default_queue() == 0;
  // CHECK: icmp eq ptr null, %{{.*}}
  // CHECK: icmp eq ptr %{{.*}}, null
}

void func(queue_t q);

void init(void) {
  queue_t q = 0;
  func(0);
  // CHECK: store ptr null, ptr %q
  // CHECK: call void @func(ptr null)
}
