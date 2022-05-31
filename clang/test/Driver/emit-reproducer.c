// RUN: rm -rf %t && mkdir %t

// RUN: echo "%s -fcrash-diagnostics-dir=%t -fsyntax-only" | sed -e 's/\\/\\\\/g' > %t.rsp

// RUN: not %clang -DFATAL @%t.rsp -gen-reproducer=off    2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DFATAL @%t.rsp -fno-crash-diagnostics 2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DFATAL @%t.rsp                        2>&1 | FileCheck %s
// RUN: not %clang -DFATAL @%t.rsp -gen-reproducer=crash  2>&1 | FileCheck %s
// RUN: not %clang -DFATAL @%t.rsp -gen-reproducer=error  2>&1 | FileCheck %s
// RUN: not %clang -DFATAL @%t.rsp -gen-reproducer=always 2>&1 | FileCheck %s
// RUN: not %clang -DFATAL @%t.rsp -gen-reproducer        2>&1 | FileCheck %s

// RUN: not %clang -DERROR @%t.rsp -gen-reproducer=off    2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DERROR @%t.rsp -fno-crash-diagnostics 2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DERROR @%t.rsp                        2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DERROR @%t.rsp -gen-reproducer=crash  2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DERROR @%t.rsp -gen-reproducer=error  2>&1 | FileCheck %s
// RUN: not %clang -DERROR @%t.rsp -gen-reproducer=always 2>&1 | FileCheck %s
// RUN: not %clang -DERROR @%t.rsp -gen-reproducer        2>&1 | FileCheck %s

// RUN:     %clang         @%t.rsp -gen-reproducer=off    2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN:     %clang         @%t.rsp -fno-crash-diagnostics 2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN:     %clang         @%t.rsp                        2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN:     %clang         @%t.rsp -gen-reproducer=crash  2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN:     %clang         @%t.rsp -gen-reproducer=error  2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN: not %clang         @%t.rsp -gen-reproducer=always 2>&1 | FileCheck %s
// RUN: not %clang         @%t.rsp -gen-reproducer        2>&1 | FileCheck %s

// RUN: not %clang -gen-reproducer=badvalue 2>&1 | FileCheck %s --check-prefix=BAD-VALUE
// BAD-VALUE: Unknown value for -gen-reproducer=: 'badvalue'

// CHECK:   note: diagnostic msg: {{.*}}emit-reproducer-{{.*}}.c
// NOT-NOT: note: diagnostic msg: {{.*}}emit-reproducer-{{.*}}.c

#ifdef FATAL
#pragma clang __debug crash
#elif ERROR
int main
#else
int main() {}
#endif
