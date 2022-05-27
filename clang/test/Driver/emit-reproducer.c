// RUN: rm -rf %t && mkdir %t

// RUN: not %clang -DFATAL %s -fcrash-diagnostics-dir=%t -gen-reproducer=off    2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DFATAL %s -fcrash-diagnostics-dir=%t -fno-crash-diagnostics 2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DFATAL %s -fcrash-diagnostics-dir=%t                        2>&1 | FileCheck %s
// RUN: not %clang -DFATAL %s -fcrash-diagnostics-dir=%t -gen-reproducer=crash  2>&1 | FileCheck %s
// RUN: not %clang -DFATAL %s -fcrash-diagnostics-dir=%t -gen-reproducer=error  2>&1 | FileCheck %s
// RUN: not %clang -DFATAL %s -fcrash-diagnostics-dir=%t -gen-reproducer=always 2>&1 | FileCheck %s
// RUN: not %clang -DFATAL %s -fcrash-diagnostics-dir=%t -gen-reproducer        2>&1 | FileCheck %s

// RUN: not %clang -DERROR %s -fcrash-diagnostics-dir=%t -gen-reproducer=off    2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DERROR %s -fcrash-diagnostics-dir=%t -fno-crash-diagnostics 2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DERROR %s -fcrash-diagnostics-dir=%t                        2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DERROR %s -fcrash-diagnostics-dir=%t -gen-reproducer=crash  2>&1 | FileCheck %s --check-prefix=NOT
// RUN: not %clang -DERROR %s -fcrash-diagnostics-dir=%t -gen-reproducer=error  2>&1 | FileCheck %s
// RUN: not %clang -DERROR %s -fcrash-diagnostics-dir=%t -gen-reproducer=always 2>&1 | FileCheck %s
// RUN: not %clang -DERROR %s -fcrash-diagnostics-dir=%t -gen-reproducer        2>&1 | FileCheck %s

// RUN:     %clang         %s -fcrash-diagnostics-dir=%t -gen-reproducer=off    2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN:     %clang         %s -fcrash-diagnostics-dir=%t -fno-crash-diagnostics 2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN:     %clang         %s -fcrash-diagnostics-dir=%t                        2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN:     %clang         %s -fcrash-diagnostics-dir=%t -gen-reproducer=crash  2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN:     %clang         %s -fcrash-diagnostics-dir=%t -gen-reproducer=error  2>&1 | FileCheck %s --check-prefix=NOT --allow-empty
// RUN: not %clang         %s -fcrash-diagnostics-dir=%t -gen-reproducer=always 2>&1 | FileCheck %s
// RUN: not %clang         %s -fcrash-diagnostics-dir=%t -gen-reproducer        2>&1 | FileCheck %s

// RUN: not %clang $s -gen-reproducer=badvalue 2>&1 | FileCheck %s --check-prefix=BAD-VALUE
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
