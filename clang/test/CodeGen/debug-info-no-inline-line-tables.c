// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-windows-msvc -debug-info-kind=limited \
// RUN:   -gno-inline-line-tables -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -gno-inline-line-tables -S -emit-llvm -o - %s | FileCheck %s \
// RUN:   -check-prefix=INLINE-FLAG
// RUN: %clang -S -emit-llvm -o - %s | FileCheck %s -check-prefix=NO-FLAG

int x;
__attribute((always_inline)) void f(void) {
  x += 1;
}
int main(void) {
  f();
  x += 2;
  return x;
}

// Check that clang emits the location of the call site and not the inlined
// function in the debug info.
// CHECK: define dso_local i32 @main()
// CHECK: %{{.+}} = load i32, i32* @x, align 4, !dbg [[DbgLoc:![0-9]+]]

// Check that the no-inline-line-tables attribute is added.
// CHECK: attributes #0 = {{.*}}"no-inline-line-tables"{{.*}}
// CHECK: attributes #1 = {{.*}}"no-inline-line-tables"{{.*}}

// CHECK: [[DbgLoc]] = !DILocation(line: 12,
// CHECK-NOT:  inlinedAt:

// INLINE-FLAG: attributes #0 = {{.*}}"no-inline-line-tables"{{.*}}
// INLINE-FLAG: attributes #1 = {{.*}}"no-inline-line-tables"{{.*}}
// NO-FLAG-NOT: "no-inline-line-tables"
