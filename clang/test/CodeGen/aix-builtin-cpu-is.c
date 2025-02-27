// RUN: echo "int main() { return __builtin_cpu_is(\"ppc970\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"ppc-cell-be\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"ppca2\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"ppc405\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"ppc440\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"ppc464\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"ppc476\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"power4\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"power5\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"power5+\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"power6\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"power6x\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s

// RUN: echo "int main() { return __builtin_cpu_is(\"power7\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DVALUE=32768 \
// RUN:   --check-prefix=CHECKOP

// RUN: echo "int main() { return __builtin_cpu_is(\"pwr7\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DVALUE=32768 \
// RUN:   --check-prefix=CHECKOP

// RUN: echo "int main() { return __builtin_cpu_is(\"power8\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DVALUE=65536 \
// RUN:   --check-prefix=CHECKOP

// RUN: echo "int main() { return __builtin_cpu_is(\"power9\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DVALUE=131072\
// RUN:   --check-prefix=CHECKOP

// RUN: echo "int main() { return __builtin_cpu_is(\"power10\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DVALUE=262144 \
// RUN:   --check-prefix=CHECKOP

// RUN: echo "int main() { return __builtin_cpu_is(\"pwr10\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DVALUE=262144 \
// RUN:   --check-prefix=CHECKOP

// RUN: echo "int main() { return __builtin_cpu_is(\"power11\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DVALUE=524288 \
// RUN:   --check-prefix=CHECKOP

// CHECK:     define i32 @main() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %retval = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %retval, align 4
// CHECK-NEXT:   ret i32 0
// CHECK-NEXT: }

// CHECKOP: @_system_configuration = external global { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i64, i32, i8, i8, i8, i8, i32, i32, i16, i16, [3 x i32], i32 }
// CHECKOP:   define i32 @main() #0 {
// CHECKOP-NEXT: entry:
// CHECKOP-NEXT:   %retval = alloca i32, align 4
// CHECKOP-NEXT:   store i32 0, ptr %retval, align 4
// CHECKOP-NEXT:   %0 = load i32, ptr getelementptr inbounds ({ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i64, i32, i8, i8, i8, i8, i32, i32, i16, i16, [3 x i32], i32 }, ptr @_system_configuration, i32 0, i32 1), align 4
// CHECKOP-NEXT:   %1 = icmp eq i32 %0, [[VALUE]]
// CHECKOP-NEXT:  %conv = zext i1 %1 to i32
// CHECKOP-NEXT:   ret i32 %conv
// CHECKOP-NEXT: }


