// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct AA {int a, b;} x;
extern int r(void);
void a(struct AA* b) {*b = (r(), x);}

// CHECK-LABEL: @a
// CHECK: %[[ADDR:.*]] = cir.alloca {{.*}} ["b"
// CHECK: cir.store {{.*}}, %[[ADDR]]
// CHECK: %[[LOAD:.*]] = cir.load deref %[[ADDR]]
// CHECK: cir.call @r
// CHECK: %[[GADDR:.*]] = cir.get_global @x
// CHECK: cir.copy %[[GADDR]] to %[[LOAD]]