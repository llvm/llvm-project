// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

class Parent {
 public:
  int field;
};

class Child : public Parent {
 public:
  int extra_field;
};

int main(void) {
  Parent p;
  Child *c = (Child*)&p;  // Intentional error here!
  c->extra_field = 42;
// CHECK: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] thread T0
// CHECK-NEXT:  {{#0 0x[0-9a-f]* in main .*wrong_downcast_on_stack.cpp}}:[[@LINE-3]]
// CHECK: [[ADDR]] is located in stack of thread T0 at offset [[OFFSET:[0-9]+]] in frame
// CHECK-NEXT:  {{#0 0x[0-9a-f]* in main }}
// CHECK:  'p'{{.*}} <== Memory access at offset [[OFFSET]] overflows this variable
  return 0;
}

