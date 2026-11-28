// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -mem2reg -o - | FileCheck %s \
// RUN:   --implicit-check-not=cir.alloca --implicit-check-not=cir.load

void use(int);

// Parameter loaded inside `if`: slot is declared further out and only read
// inside the region, so mem2reg promotes it without flattening the CFG.
void load_enclosing(int c, int x) {
  if (c)
    use(x);
}

// CHECK-LABEL: cir.func {{.*}}@load_enclosing
// CHECK: cir.call @use(%arg1)

// `a && b` is a cir.ternary; the RHS load of `b` lives in a nested region.
void land_enclosing(int a, int b) {
  if (a && b)
    use(b);
}

// CHECK-LABEL: cir.func {{.*}}@land_enclosing
// CHECK: cir.ternary
// CHECK: cir.call @use(%arg1)
