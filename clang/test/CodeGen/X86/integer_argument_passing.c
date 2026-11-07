// Test platform defaults.
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple i386-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-sie-ps5 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-pc-windows-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-NO-ASSUME-EXTENDED
//
// Same with ABI lowering library.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple i386-linux-gnu -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple i386-pc-win32 -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-sie-ps5 -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-pc-windows-gnu -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-NO-ASSUME-EXTENDED

// Test -msmall-int-abi-compat flag
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=both %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=platform %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=clang %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=both -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=platform -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-NO-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=clang -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
//
// Check that the flag has no effect on targets with different platform ABI.
// RUN: %clang_cc1 -triple x86_64-apple-darwin -msmall-int-abi-compat=both %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-apple-darwin -msmall-int-abi-compat=platform %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-apple-darwin -msmall-int-abi-compat=clang %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-apple-darwin -msmall-int-abi-compat=both -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-apple-darwin -msmall-int-abi-compat=platform -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED
// RUN: %clang_cc1 -triple x86_64-apple-darwin -msmall-int-abi-compat=clang -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-ASSUME-EXTENDED

// CALLEE-ASSUME-EXTENDED: define{{.*}} i32 @ZeroExtend(i8 noundef zeroext %a, i16 noundef zeroext %b)
// CALLEE-NO-ASSUME-EXTENDED: define{{.*}} i32 @ZeroExtend(i8 noundef %a, i16 noundef %b)
unsigned ZeroExtend(unsigned char a, unsigned short b) {
  return a*b;
}

// CALLEE-ASSUME-EXTENDED: define{{.*}} i32 @SignExtend(i8 noundef signext %a, i16 noundef signext %b)
// CALLEE-NO-ASSUME-EXTENDED: define{{.*}} i32 @SignExtend(i8 noundef %a, i16 noundef %b)
int SignExtend(char a, short b) {
  return a*b;
}

extern void f0(unsigned char x);
extern void f1(unsigned short x);
extern void f2(char x);
extern void f3(short x);

// CALLER-EXTEND: call void @f0(i8 noundef zeroext {{.*}})
// CALLER-NO-EXTEND: call void @f0(i8 noundef {{.*}})
void caller0(unsigned char x) { f0(x); }
// CALLER-EXTEND: call void @f1(i16 noundef zeroext {{.*}})
// CALLER-NO-EXTEND: call void @f1(i16 noundef {{.*}})
void caller1(unsigned short x) { f1(x); }
// CALLER-EXTEND: call void @f2(i8 noundef signext {{.*}})
// CALLER-NO-EXTEND: call void @f2(i8 noundef {{.*}})
void caller2(char x) { f2(x); }
// CALLER-EXTEND: call void @f3(i16 noundef signext {{.*}})
// CALLER-NO-EXTEND: call void @f3(i16 noundef {{.*}})
void caller3(short x) { f3(x); }
