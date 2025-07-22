// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c++ -std=c++17 %s -debug-info-kind=line-tables-only -emit-llvm -o - -gno-column-info \
// RUN: | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - -gno-column-info \
// RUN: | FileCheck %s

// Check the goto branches get Key Instructions metadata.
void ext();
void test_goto(void) {
// CHECK: br label %dst1, !dbg [[G1R1:!.*]]
  goto dst1;
dst1:
  ext();

  void *ptr = &&dst2;
// CHECK: br label %indirectgoto, !dbg [[G3R1:!.*]]
  goto *ptr;
dst2:
  ext();

// CHECK: br label %dst3, !dbg [[G4R1:!.*]]
  goto *&&dst3;
dst3:
  ext();

  return;
}

// CHECK: [[G1R1]] = !DILocation(line: 10, scope: ![[#]], atomGroup: 1, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation(line: 16, scope: ![[#]], atomGroup: 3, atomRank: 1)
// CHECK: [[G4R1]] = !DILocation(line: 21, scope: ![[#]], atomGroup: 4, atomRank: 1)
