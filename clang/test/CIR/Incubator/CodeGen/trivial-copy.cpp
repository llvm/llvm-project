// RUN:   %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s

struct Trivial {
  int i;
};

void CopyCTor(Trivial &a) {
  Trivial b(a);
  
// CHECK:         cir.copy
// CHECK-NOT:     cir.call {{.*}}_ZN7TrivialC2ERKS_
// CHECK-NOT:     cir.func {{.*}}_ZN7TrivialC2ERKS_
}

void CopyAssign(Trivial &a) {
  Trivial b = a;
// CHECK:         cir.copy
// CHECK-NOT:     cir.call {{.*}}_ZN7TrivialaSERKS_
// CHECK-NOT:     cir.func {{.*}}_ZN7TrivialaSERKS_
}
