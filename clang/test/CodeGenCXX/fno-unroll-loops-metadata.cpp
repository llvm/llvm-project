// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s -O0 -disable-llvm-optzns -fno-unroll-loops | FileCheck --check-prefix=NO_UNROLL_MD %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s -O1 -disable-llvm-optzns -fno-unroll-loops | FileCheck --check-prefix=UNROLL_DISABLED_MD %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s -O2 -disable-llvm-optzns -fno-unroll-loops | FileCheck --check-prefix=UNROLL_DISABLED_MD %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s -O3 -disable-llvm-optzns -fno-unroll-loops | FileCheck --check-prefix=UNROLL_DISABLED_MD %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s -O3 -disable-llvm-optzns | FileCheck --check-prefix=NO_UNROLL_MD %s

// NO_UNROLL_MD-NOT: llvm.loop.unroll.disable

// Verify unroll.disable metadata is added to while loop with -fno-unroll-loops
// and optlevel > 0.
void while_test(int *List, int Length) {
  // UNROLL_DISABLED_MD: define {{.*}} @_Z10while_test
  int i = 0;

  while (i < Length) {
    // UNROLL_DISABLED_MD: br label {{.*}}, !llvm.loop [[LOOP_1:![0-9]+]]
    List[i] = i * 2;
    i++;
  }
}

// Verify unroll.disable metadata is added to do-while loop with
// -fno-unroll-loops and optlevel > 0.
void do_test(int *List, int Length) {
  // UNROLL_DISABLED_MD: define {{.*}} @_Z7do_test
  int i = 0;

  do {
    // UNROLL_DISABLED_MD: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop [[LOOP_2:![0-9]+]]
    List[i] = i * 2;
    i++;
  } while (i < Length);
}

// Verify unroll.disable metadata is added to while loop with -fno-unroll-loops
// and optlevel > 0.
void for_test(int *List, int Length) {
  // UNROLL_DISABLED_MD: define {{.*}} @_Z8for_test
  for (int i = 0; i < Length; i++) {
    // UNROLL_DISABLED_MD: br label {{.*}}, !llvm.loop [[LOOP_3:![0-9]+]]
    List[i] = i * 2;
  }
}

// UNROLL_DISABLED_MD: [[LOOP_1]] = distinct !{[[LOOP_1]], [[MP:![0-9]+]], [[UNROLL_DISABLE:![0-9]+]]}
// UNROLL_DISABLED_MD: [[MP]] = !{!"llvm.loop.mustprogress"}
// UNROLL_DISABLED_MD: [[UNROLL_DISABLE]] = !{!"llvm.loop.unroll.disable"}
// UNROLL_DISABLED_MD: [[LOOP_2]] = distinct !{[[LOOP_2]], [[MP]], [[UNROLL_DISABLE]]}
// UNROLL_DISABLED_MD: [[LOOP_3]] = distinct !{[[LOOP_3]], [[MP]], [[UNROLL_DISABLE]]}
