// RUN: %clang_cc1 -fprofile-instrument=llvm -emit-llvm %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}"

// RUN: echo -e "[llvm]\nfunction:foo=skip" > %t0.list
// RUN: %clang_cc1 -fprofile-instrument=llvm -fprofile-list=%t0.list -emit-llvm %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}" --check-prefixes=CHECK,SKIP-FOO

// RUN: echo -e "[csllvm]\nfunction:bar=forbid" > %t1.list
// RUN: %clang_cc1 -fprofile-instrument=csllvm -fprofile-list=%t1.list -emit-llvm %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}" --check-prefixes=CHECK,FORBID-BAR
// RUN: %clang_cc1 -fprofile-instrument=llvm -fprofile-list=%t1.list -emit-llvm %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}"

// RUN: echo -e "[llvm]\ndefault:forbid\nfunction:foo=allow" > %t2.list
// RUN: %clang_cc1 -fprofile-instrument=llvm -fprofile-list=%t2.list -emit-llvm %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}" --check-prefixes=CHECK,FORBID

// RUN: echo "[llvm]" > %t2.list
// RUN: echo "source:%s=forbid" | sed -e 's/\\/\\\\/g' >> %t2.list
// RUN: echo "function:foo=allow" >> %t2.list
// RUN: %clang_cc1 -fprofile-instrument=llvm -fprofile-list=%t2.list -emit-llvm %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}" --check-prefixes=CHECK,FORBID

// SKIP-FOO: skipprofile
// CHECK-LABEL: define {{.*}} @foo
int foo(int a) { return 4 * a + 1; }

// FORBID-BAR: noprofile
// FORBID: noprofile
// CHECK-LABEL: define {{.*}} @bar
int bar(int a) { return 4 * a + 2; }

// FORBID: noprofile
// CHECK-LABEL: define {{.*}} @goo
int goo(int a) { return 4 * a + 3; }
