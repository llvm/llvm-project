// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-bitfield-constant-conversion -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR

struct C {
  int c : 8;
};

void foo(void) {
  struct C c;
  c.c = 800;
}

// CIR: %[[SETBF:.*]] = cir.set_bitfield{{.*}} loc(#[[SET_LOC:loc[0-9]+]])
// CIR-DAG: #[[SET_BEGIN:loc[0-9]+]] = loc("{{.*}}bitfield-assignment-loc.c":9:3)
// CIR-DAG: #[[SET_END:loc[0-9]+]] = loc("{{.*}}bitfield-assignment-loc.c":9:9)
// CIR-DAG: #[[SET_LOC]] = loc(fused[#[[SET_BEGIN]], #[[SET_END]]])
