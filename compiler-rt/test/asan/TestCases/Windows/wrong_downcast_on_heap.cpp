// RUN: %clang_cl_asan %Od %s %Fe%t
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
  Parent *p = new Parent;
  Child *c = (Child*)p;  // Intentional error here!
  c->extra_field = 42;
// CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] thread T0
// CHECK:   {{#0 0x[0-9a-f]* in main .*wrong_downcast_on_heap.cpp}}:[[@LINE-3]]
// CHECK: [[ADDR]] is located 0 bytes after 4-byte region
// CHECK: allocated by thread T0 here:
// CHECK:   #0 {{.*}} operator new
  return 0;
}

