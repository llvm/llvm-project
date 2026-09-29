// RUN: %clang_cc1       -triple x86_64 %s                                         -emit-llvm -o - | FileCheck --check-prefixes=X86,CHECK %s
// RUN: %clang_cc1       -triple x86_64 %s -fexperimental-new-constant-interpreter -emit-llvm -o - | FileCheck --check-prefixes=X86,CHECK %s
// RUN: %clang_cc1       -triple avr    %s                                         -emit-llvm -o - | FileCheck --check-prefixes=AVR,CHECK %s
// RUN: %clang_cc1       -triple avr    %s -fexperimental-new-constant-interpreter -emit-llvm -o - | FileCheck --check-prefixes=AVR,CHECK %s

// RUN: %clang_cc1 -xc++ -triple x86_64 %s                                         -emit-llvm -o - | FileCheck --check-prefixes=X86,CHECK %s
// RUN: %clang_cc1 -xc++ -triple x86_64 %s -fexperimental-new-constant-interpreter -emit-llvm -o - | FileCheck --check-prefixes=X86,CHECK %s
// RUN: %clang_cc1 -xc++ -triple avr    %s                                         -emit-llvm -o - | FileCheck --check-prefixes=AVR,CHECK %s
// RUN: %clang_cc1 -xc++ -triple avr    %s -fexperimental-new-constant-interpreter -emit-llvm -o - | FileCheck --check-prefixes=AVR,CHECK %s

int a;
__UINTPTR_TYPE__ ptrasintadd1 = (__UINTPTR_TYPE__)&a - 4;
__UINTPTR_TYPE__ ptrasintadd2 = (__UINTPTR_TYPE__)&a + 4;
__UINTPTR_TYPE__ ptrasintadd3 = ((__UINTPTR_TYPE__)&a + 4) + 10;
__UINTPTR_TYPE__ ptrasintadd4 = (__UINTPTR_TYPE__)&a + ((__UINTPTR_TYPE__)-1);
__UINTPTR_TYPE__ ptrasintadd5 = 4 + (__UINTPTR_TYPE__)&a;
__UINTPTR_TYPE__ ptrasintadd6 = 10 + ((__UINTPTR_TYPE__)&a + 4);

// CHECK: @ptrasintadd1 = global {{.*}} ptrtoint (ptr getelementptr (i8, ptr @a, {{.*}} -4) to {{.*}})
// CHECK: @ptrasintadd2 = global {{.*}} ptrtoint (ptr getelementptr (i8, ptr @a, {{.*}} 4) to {{.*}})
// CHECK: @ptrasintadd3 = global {{.*}} ptrtoint (ptr getelementptr (i8, ptr @a, {{.*}} 14) to {{.*}})
// AVR:   @ptrasintadd4 = global {{.*}} ptrtoint (ptr getelementptr (i8, ptr @a, {{.*}} 65535) to {{.*}})
// X86:   @ptrasintadd4 = global {{.*}} ptrtoint (ptr getelementptr (i8, ptr @a, {{.*}} -1) to {{.*}})
// CHECK: @ptrasintadd5 = global {{.*}} ptrtoint (ptr getelementptr (i8, ptr @a, {{.*}} 4) to {{.*}})
// CHECK: @ptrasintadd6 = global {{.*}} ptrtoint (ptr getelementptr (i8, ptr @a, {{.*}} 14) to {{.*}})
