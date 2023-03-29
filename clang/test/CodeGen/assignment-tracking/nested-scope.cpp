// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone -O0 \
// RUN:     -emit-llvm  -fexperimental-assignment-tracking=forced %s -o -        \
// RUN:     -disable-O0-optnone                                                  \
// RUN: | FileCheck %s

// Check that dbg.assign intrinsics get a !dbg with with the same scope as
// their variable.

// CHECK: call void @llvm.dbg.assign({{.+}}, metadata [[local:![0-9]+]], {{.+}}, {{.+}}, {{.+}}), !dbg [[dbg:![0-9]+]]
// CHECK-DAG: [[local]] = !DILocalVariable(name: "local", scope: [[scope:![0-9]+]],
// CHECK-DAG: [[dbg]] = !DILocation({{.+}}, scope: [[scope]])
// CHECK-DAG: [[scope]] = distinct !DILexicalBlock

void ext(int*);
void fun() {
  {
    int local;
  }
}

