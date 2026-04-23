// RUN: %clangxx_tysan %s -o %t && %run %t 10 >%t.out.0 2>&1
// RUN: FileCheck %s < %t.out.0

namespace fancy_namespace {
struct fancy_struct {
  int member;
};
} // namespace fancy_namespace

int main() {
  fancy_namespace::fancy_struct *x = new fancy_namespace::fancy_struct{42};
  *(float *)&x = 1.0f;
  // CHECK: WRITE of size 4 at {{.*}} with type float accesses an existing object of type fancy_namespace::fancy_struct*

  fancy_namespace::fancy_struct **double_indirection = &x;
  *(float *)&double_indirection = 1.0f;
  // CHECK: WRITE of size 4 at {{.*}} with type float accesses an existing object of type fancy_namespace::fancy_struct**
}
