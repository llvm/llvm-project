// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Simple scalar return: __retval should be eliminated.
int add(int a, int b) { return a + b; }

// CHECK-LABEL: cir.func{{.*}} @add(
// CHECK-NOT:     ["__retval"]
// CHECK:         cir.return
// CHECK:       }

// Void return: no __retval at all.
void noop(void) {}

// CHECK-LABEL: cir.func{{.*}} @noop(
// CHECK-NOT:     ["__retval"]
// CHECK:         cir.return
// CHECK:       }

int select_val(int a, int b, int c) {
  if (c) return a;
  return b;
}

// CHECK-LABEL: cir.func{{.*}} @select_val(
// CHECK-NOT:     ["__retval"]
// CHECK:         cir.return
// CHECK:       }

typedef struct { int x; int y; } Pair;
Pair make_pair(int a, int b) {
  Pair p = {a, b};
  return p;
}

// CHECK-LABEL: cir.func{{.*}} @make_pair(
// CHECK:         cir.alloca !rec_Pair, !cir.ptr<!rec_Pair>, ["__retval", init]
// CHECK:         cir.load %{{.+}} : !cir.ptr<!rec_Pair>, !rec_Pair
// CHECK:         cir.return
// CHECK:       }
