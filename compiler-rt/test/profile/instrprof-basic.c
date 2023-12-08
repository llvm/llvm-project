// RUN: %clang_profgen -o %t -O3 %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s --check-prefix=COMMON --check-prefix=ORIG
//
// RUN: rm -fr %t.dir1
// RUN: mkdir -p %t.dir1
// RUN: env LLVM_PROFILE_FILE=%t.dir1/profraw_e_%1m %run %t
// RUN: env LLVM_PROFILE_FILE=%t.dir1/profraw_e_%1m %run %t
// RUN: llvm-profdata merge -o %t.em.profdata %t.dir1
// RUN: %clang_profuse=%t.em.profdata -o - -S -emit-llvm %s | FileCheck %s --check-prefix=COMMON --check-prefix=MERGE
//
// RUN: rm -fr %t.dir2
// RUN: mkdir -p %t.dir2
// RUN: %clang_profgen=%t.dir2/%m.profraw -o %t.merge -O3 %s
// RUN: %run %t.merge
// RUN: %run %t.merge
// RUN: llvm-profdata merge -o %t.m.profdata %t.dir2/
// RUN: %clang_profuse=%t.m.profdata -o - -S -emit-llvm %s | FileCheck %s --check-prefix=COMMON --check-prefix=MERGE
//
// Test that merging is enabled by default with -fprofile-generate=
// RUN: rm -fr %t.dir3
// RUN: mkdir -p %t.dir3
// RUN: %clang_pgogen=%t.dir3/ -o %t.merge3 -O0 %s
// RUN: %run %t.merge3
// RUN: %run %t.merge3
// RUN: %run %t.merge3
// RUN: %run %t.merge3
// RUN: llvm-profdata merge -o %t.m3.profdata %t.dir3/
// RUN: %clang_profuse=%t.m3.profdata -O0 -o - -S -emit-llvm %s | FileCheck %s --check-prefix=COMMON --check-prefix=PGOMERGE
//
// Test that merging is enabled by default with -fprofile-generate
// RUN: rm -fr %t.dir4
// RUN: mkdir -p %t.dir4
// RUN: %clang_pgogen -o %t.dir4/merge4 -O0 %s
// RUN: cd %t.dir4
// RUN: %run %t.dir4/merge4
// RUN: %run %t.dir4/merge4
// RUN: %run %t.dir4/merge4
// RUN: %run %t.dir4/merge4
// RUN: rm -f %t.dir4/merge4*
// RUN: llvm-profdata merge -o %t.m4.profdata ./
// RUN: %clang_profuse=%t.m4.profdata -O0 -o - -S -emit-llvm %s | FileCheck %s --check-prefix=COMMON  --check-prefix=PGOMERGE

/// Test that the merge pool size can be larger than 10.
// RUN: rm -fr %t.dir5
// RUN: mkdir -p %t.dir5
// RUN: env LLVM_PROFILE_FILE=%t.dir5/e_%20m.profraw %run %t
// RUN: not ls %t.dir5/e_%20m.profraw
// RUN: ls %t.dir5/e_*.profraw | count 1

int begin(int i) {
  // COMMON: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  if (i)
    return 0;
  return 1;
}

int end(int i) {
  // COMMON: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD2:[0-9]+]]
  if (i)
    return 0;
  return 1;
}

int main(int argc, const char *argv[]) {
  begin(0);
  end(1);

  // COMMON: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD2:[0-9]+]]
  if (argc)
    return 0;
  return 1;
}

// ORIG: ![[PD1]] = !{!"branch_weights", i32 1, i32 2}
// ORIG: ![[PD2]] = !{!"branch_weights", i32 2, i32 1}
// MERGE: ![[PD1]] = !{!"branch_weights", i32 1, i32 3}
// MERGE: ![[PD2]] = !{!"branch_weights", i32 3, i32 1}
// PGOMERGE: ![[PD1]] = !{!"branch_weights", i32 0, i32 4}
// PGOMERGE: ![[PD2]] = !{!"branch_weights", i32 4, i32 0}
