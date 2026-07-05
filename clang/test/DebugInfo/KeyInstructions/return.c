// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-CXX

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s

// Check the stores to `retval` allocas and branches to `return` block are in
// the same atom group. They are both rank 1, which could in theory introduce
// an extra step in some optimized code. This low risk currently feels an
// acceptable for keeping the code a bit simpler (as opposed to adding
// scaffolding to make the store rank 2).

// Also check that in the case of a single return (no control flow) the
// return instruction inherits the atom group of the branch to the return
// block when the blocks get folded togather.

#ifdef __cplusplus
#define nomangle extern "C"
#else
#define nomangle
#endif

int g;
nomangle float a() {
// CHECK: float @a()
  if (g)
// CHECK: if.then:
// CHECK-NEXT: %1 = load i32, ptr @g{{.*}}, !dbg [[G2R3:!.*]]
// CHECK-NEXT: %conv = sitofp i32 %1 to float{{.*}}, !dbg [[G2R2:!.*]]
// CHECK-NEXT: store float %conv, ptr %retval{{.*}}, !dbg [[G2R1:!.*]]
// CHECK-NEXT: br label %return{{.*}}, !dbg [[G2R1]]
    return g;
// CHECK: if.end:
// CHECK-NEXT: store float 1.000000e+00, ptr %retval{{.*}}, !dbg [[G3R1:!.*]]
// CHECK-NEXT: br label %return, !dbg [[G3R1]]

// CHECK: return:
// CHECK-NEXT:  %2 = load float, ptr %retval{{.*}}, !dbg [[G4R2:!.*]]
// CHECK-NEXT:  ret float %2{{.*}}, !dbg [[G4R1:!.*]]
  return 1;
}

// CHECK: void @b()
// CHECK: ret void{{.*}}, !dbg [[B_G1R1:!.*]]
nomangle void b() { return; }

// CHECK: i32 @c()
// CHECK: %add = add{{.*}}, !dbg [[C_G1R2:!.*]]
// CHECK: ret i32 %add{{.*}}, !dbg [[C_G1R1:!.*]]
nomangle int c() { return g + 1; }

// NOTE: (return) (g = 1) are two separate atoms.
// CHECK: i32 @d()
// CHECK: store{{.*}}, !dbg [[D_G2R1:!.*]]
// CHECK: ret i32 1{{.*}}, !dbg [[D_G1R1:!.*]]
nomangle int d() { return g = 1; }

// The implicit return here get the line number of the closing brace; make it
// key to match existing behaviour.
// CHECK: void @e()
// CHECK: ret void, !dbg [[E_G1R1:!.*]]
nomangle void e() {}

#ifdef __cplusplus
// CHECK-CXX: ptr @_Z1fRi
int &f(int &r) {
// Include ctrl-flow to stop ret value store being elided.
    if (r)
// CHECK-CXX: if.then:
// CHECK-CXX-NEXT: %2 = load ptr, ptr %r.addr{{.*}}, !dbg [[F_G2R2:!.*]], !nonnull
// CHECK-CXX-NEXT: store ptr %2, ptr %retval{{.*}}, !dbg [[F_G2R1:!.*]]
// CHECK-CXX-NEXT: br label %return, !dbg [[F_G2R1:!.*]]
    return r;

// CHECK-CXX: if.end:
// CHECK-CXX-NEXT: store ptr @g, ptr %retval{{.*}}, !dbg [[F_G3R1:!.*]]
// CHECK-CXX-NEXT: br label %return, !dbg [[F_G3R1:!.*]]
// CHECK-CXX: return:
// CHECK-CXX-NEXT: %3 = load ptr, ptr %retval{{.*}}, !dbg [[F_G4R2:!.*]]
// CHECK-CXX-NEXT: ret ptr %3, !dbg [[F_G4R1:!.*]]
  return g;
}
#endif

// CHECK: [[G2R3]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 3)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[B_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[C_G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[C_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[D_G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[D_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[E_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK-CXX: [[F_G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK-CXX: [[F_G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK-CXX: [[F_G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK-CXX: [[F_G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK-CXX: [[F_G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
