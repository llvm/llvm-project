// RUN: %clang_cc1 -debug-info-kind=limited -gno-column-info -std=c++11 -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap %s
// PR19864
extern int v[2];
int a = 0, b = 0;
int main() {
#line 100
  for (int x : v)
    if (x)
      ++b; // CHECK: add nsw{{.*}}, 1
    else
      ++a; // CHECK: add nsw{{.*}}, 1
  // The continuation block if the if statement should not share the
  // location of the ++a statement. The branch back to the start of the loop
  // should be attributed to the loop header line.

  // CHECK: br label
  // CHECK: br label
  // CHECK: br label {{.*}}, !dbg [[DBG1:![0-9]*]], !llvm.loop [[L1:![0-9]*]]

#line 200
  while (a)
    if (b)
      ++b; // CHECK: add nsw{{.*}}, 1
    else
      ++a; // CHECK: add nsw{{.*}}, 1

  // CHECK: br label
  // CHECK: br label {{.*}}, !dbg [[DBG2:![0-9]*]], !llvm.loop [[L2:![0-9]*]]

#line 300
  for (; a; )
    if (b)
      ++b; // CHECK: add nsw{{.*}}, 1
    else
      ++a; // CHECK: add nsw{{.*}}, 1

  // CHECK: br label
  // CHECK: br label {{.*}}, !dbg [[DBG3:![0-9]*]], !llvm.loop [[L3:![0-9]*]]

#line 400
  int x[] = {1, 2};
  for (int y : x)
    if (b)
      ++b; // CHECK: add nsw{{.*}}, 1
    else
      ++a; // CHECK: add nsw{{.*}}, 1

  // CHECK: br label
  // CHECK: br label {{.*}}, !dbg [[DBG4:![0-9]*]], !llvm.loop [[L4:![0-9]*]]

  // CHECK-DAG: [[DBG1]] = !DILocation(line: 100, scope: !{{.*}})
  // CHECK-DAG: [[DBG2]] = !DILocation(line: 200, scope: !{{.*}})
  // CHECK-DAG: [[DBG3]] = !DILocation(line: 300, scope: !{{.*}})
  // CHECK-DAG: [[DBG4]] = !DILocation(line: 401, scope: !{{.*}})

  // CHECK-DAG: [[L1]] = distinct !{[[L1]], [[SLDBG1:![0-9]*]], [[ELDBG1:![0-9]*]]}
  // CHECK-DAG: [[SLDBG1]] = !DILocation(line: 100, scope: !{{.*}})
  // CHECK-DAG: [[ELDBG1]] = !DILocation(line: 104, scope: !{{.*}})

  // CHECK-DAG: [[L2]] = distinct !{[[L2]], [[SLDBG2:![0-9]*]], [[ELDBG2:![0-9]*]], [[MP:![0-9]+]]}
  // CHECK-DAG: [[SLDBG2]] = !DILocation(line: 200, scope: !{{.*}})
  // CHECK-DAG: [[ELDBG2]] = !DILocation(line: 204, scope: !{{.*}})

  // CHECK-DAG: [[L3]] = distinct !{[[L3]], [[SLDBG3:![0-9]*]], [[ELDBG3:![0-9]*]], [[MP]]}
  // CHECK-DAG: [[SLDBG3]] = !DILocation(line: 300, scope: !{{.*}})
  // CHECK-DAG: [[ELDBG3]] = !DILocation(line: 304, scope: !{{.*}})

  // CHECK-DAG: [[L4]] = distinct !{[[L4]], [[SLDBG4:![0-9]*]], [[ELDBG4:![0-9]*]]}
  // CHECK-DAG: [[SLDBG4]] = !DILocation(line: 401, scope: !{{.*}})
  // CHECK-DAG: [[ELDBG4]] = !DILocation(line: 405, scope: !{{.*}})
}
