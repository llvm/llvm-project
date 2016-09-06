// RUN: rm -fr %t.prof
// RUN: %clang_pgogen=%t.prof/ -o %t.gen.cs -O2 %s
// RUN: %t.gen.cs
// RUN: llvm-profdata merge -o %t.cs.profdata %t.prof/
// Check context sensitive profile
// RUN: %clang_profuse=%t.cs.profdata  -O2 -emit-llvm -S %s -o - | FileCheck %s --check-prefix=CS
//
// RUN: %clang_profgen=%t.profraw -o %t.gen.cis -O2 %s
// RUN: %t.gen.cis
// RUN: llvm-profdata merge -o %t.cis.profdata %t.profraw
// Check context insenstive profile
// RUN: %clang_profuse=%t.cis.profdata  -O2 -emit-llvm -S %s -o - | FileCheck %s --check-prefix=CIS
int g1 = 1;
int g2 = 2;
static void toggle(int t) {
  if (t & 1)
    g1 *= t;
  else
    g2 *= t;
}

int main() {
  int i;
  // CS: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  // CIS: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD:[0-9]+]]
  toggle(g1);
  // CS: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD2:[0-9]+]]
  // CIS: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD:[0-9]+]]
  toggle(g2);
  return 0;
}

// CS: ![[PD1]] = !{!"branch_weights", i32 0, i32 1}
// CS: ![[PD2]] = !{!"branch_weights", i32 1, i32 0}
// CIS: ![[PD]] = !{!"branch_weights", i32 2, i32 2}
