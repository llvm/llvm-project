// RUN: %clang_cc1 -O2 -triple -x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -triple -i386-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -triple -i386-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=none -triple -x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=none -triple -i386-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=none -triple -i386-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-NO-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=conservative -triple -x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=conservative -triple -i386-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=conservative -triple -i386-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=assumed -triple -x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=assumed -triple -i386-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-EXTEND
// RUN: %clang_cc1 -O2 -mextend-small-integers=assumed -triple -i386-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-EXTEND
// TODO: enable the belowing tests when 'default' is supported.
// %clang_cc1 -O2 -mextend-small-integers=default -triple -x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-EXTEND
// %clang_cc1 -O2 -mextend-small-integers=default -triple -i386-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-EXTEND
// %clang_cc1 -O2 -mextend-small-integers=default -triple -i386-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CALLER-EXTEND,CALLEE-NO-EXTEND


// CALLEE-EXTEND: define{{.*}} i32 @ZeroExtend(i8 noundef %a, i16 noundef %b)
// CALLEE-NO-EXTEND: define{{.*}} i32 @ZeroExtend(i8 noundef zeroext %a, i16 noundef zeroext %b)
unsigned ZeroExtend(unsigned char a, unsigned short b) {
  return a*b;
}

// CALLEE-EXTEND: define{{.*}} i32 @SignExtend(i8 noundef %a, i16 noundef %b)
// CALLEE-NO-EXTEND: define{{.*}} i32 @SignExtend(i8 noundef signext %a, i16 noundef signext %b)
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
