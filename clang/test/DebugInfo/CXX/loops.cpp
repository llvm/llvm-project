// RUN: %clang -g -gcolumn-info -std=c++11 -S -emit-llvm %s -o - | FileCheck %s
extern int v[2];
int a = 0, b = 0;
int main() {
#line 100
  for (int x : v) {
    if (x)
      ++b;
    else
      ++a;
  }

  // CHECK: br label {{.*}}, !dbg [[DBG1:![0-9]*]], !llvm.loop [[L1:![0-9]*]]

#line 200
  while (a)
    if (b)
      ++b;
    else
      ++a;
  // CHECK: br label {{.*}}, !dbg [[DBG2:![0-9]*]], !llvm.loop [[L2:![0-9]*]]

#line 300
  for (unsigned i = 0; i < 100; i++) {
    a++;
    for (int y : v)
      ++b;
  }

  // CHECK: br label {{.*}}, !dbg [[DBG3:![0-9]*]], !llvm.loop [[L3:![0-9]*]]
  // CHECK: br label {{.*}}, !dbg [[DBG3:![0-9]*]], !llvm.loop [[L4:![0-9]*]]


  // CHECK-DAG: [[L1]] = distinct !{[[L1]], [[SLDBG1:![0-9]*]], [[ELDBG1:![0-9]*]]}
  // CHECK-DAG: [[SLDBG1]] = !DILocation(line: 100, column: 3, scope: !{{.*}})
  // CHECK-DAG: [[ELDBG1]] = !DILocation(line: 105, column: 3, scope: !{{.*}})

  // CHECK-DAG: [[L2]] = distinct !{[[L2]], [[SLDBG2:![0-9]*]], [[ELDBG2:![0-9]*]], [[MP:![0-9]+]]}
  // CHECK-DAG: [[SLDBG2]] = !DILocation(line: 200, column: 3, scope: !{{.*}})
  // CHECK-DAG: [[ELDBG2]] = !DILocation(line: 204, column: 9, scope: !{{.*}})

  // CHECK-DAG: [[L3]] = distinct !{[[L3]], [[SLDBG3:![0-9]*]], [[ELDBG3:![0-9]*]]}
  // CHECK-DAG: [[SLDBG3]] = !DILocation(line: 302, column: 5, scope: !{{.*}})
  // CHECK-DAG: [[ELDBG3]] = !DILocation(line: 303, column: 9, scope: !{{.*}})
  //
  // CHECK-DAG: [[L4]] = distinct !{[[L4]], [[SLDBG4:![0-9]*]], [[ELDBG4:![0-9]*]], [[MP]]}
  // CHECK-DAG: [[SLDBG4]] = !DILocation(line: 300, column: 3, scope: !{{.*}})
  // CHECK-DAG: [[ELDBG4]] = !DILocation(line: 304, column: 3, scope: !{{.*}})
}
