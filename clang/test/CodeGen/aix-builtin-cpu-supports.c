// RUN: echo "int main() { return __builtin_cpu_supports(\"4xxmac\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"altivec\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=46 -DOP=ugt -DBIT=i32 -DVALUE=0 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"archpmu\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"booke\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"cellbe\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"darn\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=uge -DBIT=i32 -DVALUE=131072 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"dscr\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=uge -DBIT=i32 -DVALUE=65536 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"ebb\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=uge -DBIT=i32 -DVALUE=65536 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"efpdouble\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"efpsingle\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"pa6t\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"fpu\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"htm\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=ugt -DLABLE=59  -DBIT=i64 -DVALUE=0 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCALL

// RUN: echo "int main() { return __builtin_cpu_supports(\"htm-nosc\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"htm-no-suspend\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"ic_snoop\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"isel\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"mma\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=ugt -DLABLE=62 -DBIT=i64 -DVALUE=0 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCALL

// RUN: echo "int main() { return __builtin_cpu_supports(\"mmu\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"notb\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"arch_2_05\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"arch_2_06\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=uge -DBIT=i32 -DVALUE=32768 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"arch_2_07\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=uge -DBIT=i32 -DVALUE=65536 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"arch_3_00\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=uge -DBIT=i32 -DVALUE=131072 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"arch_3_1\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=uge -DBIT=i32 -DVALUE=262144 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"dfp\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=53 -DOP=ne -DBIT=i32 -DVALUE=0 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"power4\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"power5\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"power5+\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"power6x\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"ppc32\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"ppc601\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"ppc64\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"ppcle\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"smt\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=44 -DMASK=3 -DOP=eq -DBIT=i32 -DVALUE=3 \
// RUN:   --check-prefixes=CHECKOP,OPMASK,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"spe\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"scv\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"tar\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=1 -DOP=uge -DBIT=i32 -DVALUE=65536 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"true_le\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=1 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"ucache\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=5 -DMASK=2 -DOP=eq -DBIT=i32 -DVALUE=2 \
// RUN:   --check-prefixes=CHECKOP,OPMASK,SYSCONF

// RUN: echo "int main() { return __builtin_cpu_supports(\"vcrypto\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck -DBOOL=0 %s

// RUN: echo "int main() { return __builtin_cpu_supports(\"vsx\");}" > %t.c
// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %t.c | FileCheck %s -DPOS=46 -DOP=ugt -DBIT=i32 -DVALUE=1 \
// RUN:   --check-prefixes=CHECKOP,OPRT,SYSCONF

// CHECK:     define i32 @main() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %retval = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %retval, align 4
// CHECK-NEXT:   ret i32 [[BOOL]]
// CHECK-NEXT: }

// SYSCONF: @_system_configuration = external global { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i64, i32, i8, i8, i8, i8, i32, i32, i16, i16, [3 x i32], i32 }

// CHECKOP:   define i32 @main() #0 {
// CHECKOP-NEXT: entry:
// CHECKOP-NEXT:   %retval = alloca i32, align 4
// CHECKOP-NEXT:   store i32 0, ptr %retval, align 4

// SYSCONF-NEXT:   %0 = load i32, ptr getelementptr inbounds ({ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i64, i32, i8, i8, i8, i8, i32, i32, i16, i16, [3 x i32], i32 }, ptr @_system_configuration, i32 0, i32 [[POS]]), align 4
// SYSCALL-NEXT:  %0 = call i64 @getsystemcfg(i32 [[LABLE]])

// OPRT-NEXT:  %1 = icmp [[OP]] [[BIT]] %0, [[VALUE]]
// OPRT-NEXT:     %conv = zext i1 %1 to i32

// OPMASK-NEXT:  %1 = and i32 %0, [[MASK]]
// OPMASK-NEXT:  %2 = icmp [[OP]] i32 %1, [[VALUE]]
// OPMASK-NEXT:  %conv = zext i1 %2 to i32

// CHECKOP-NEXT:   ret i32 %conv
// CHECKOP-NEXT: }

// SYSCALL: declare i64 @getsystemcfg(i32)


