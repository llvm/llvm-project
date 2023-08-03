// RUN: rm -rf %t.idx
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -mmacosx-version-min=10.7 -x c-header %S/Inputs/head.h -o %t.h.pch -index-store-path %t.idx
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -mmacosx-version-min=10.7 -c %s -o %t.o -index-store-path %t.idx -include %t.h -Werror
// RUN: c-index-test core -print-unit %t.idx | FileCheck %s

int main() {
  test1_func();
}

// CHECK: print-units-with-pch.c.tmp.h.pch
// CHECK: is-system: 0
// CHECK: has-main: 0
// CHECK: main-path: {{$}}
// CHECK: out-file: {{.*}}{{/|\\}}print-units-with-pch.c.tmp.h.pch
// CHECK: DEPEND START
// CHECK: Record | user | {{.*}}{{/|\\}}Inputs{{/|\\}}head.h | head.h
// CHECK: DEPEND END (1)

// CHECK: print-units-with-pch.c.tmp.o
// CHECK: is-system: 0
// CHECK: has-main: 1
// CHECK: main-path: {{.*}}{{/|\\}}print-units-with-pch.c
// CHECK: out-file: {{.*}}{{/|\\}}print-units-with-pch.c.tmp.o
// CHECK: DEPEND START
// CHECK: Unit | user | {{.*}}{{/|\\}}print-units-with-pch.c.tmp.h.pch | print-units-with-pch.c.tmp.h.pch
// CHECK: Record | user | {{.*}}{{/|\\}}print-units-with-pch.c | print-units-with-pch.c
// CHECK: DEPEND END (2)
