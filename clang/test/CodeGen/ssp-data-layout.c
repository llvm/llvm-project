// RUN: %clang_cc1 -triple x86_64-linux-gnu    -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP
// RUN: %clang_cc1 -triple x86_64-linux-gnu    -emit-llvm -o - %s -stack-protector 2 | FileCheck %s --check-prefix=STRONG
// RUN: %clang_cc1 -triple x86_64-linux-gnu    -emit-llvm -o - %s -stack-protector 3 | FileCheck %s --check-prefix=STRONG
// RUN: %clang_cc1 -triple x86_64-linux-gnu    -emit-llvm -o - %s -stack-protector 1 -stack-protector-buffer-size 33 | FileCheck %s --check-prefix=SSP33
// RUN: %clang_cc1 -triple x86_64-linux-gnu    -emit-llvm -o - %s -stack-protector 1 -stack-protector-buffer-size 5  | FileCheck %s --check-prefix=SSP5
// Non-Darwin platforms share the same ssp.protected emission rules as Linux:
// RUN: %clang_cc1 -triple i386-pc-linux-gnu   -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP
// RUN: %clang_cc1 -triple amd64-pc-openbsd    -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP
// RUN: %clang_cc1 -triple i386-pc-windows-msvc -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP
// RUN: %clang_cc1 -triple x86_64-w64-mingw32  -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP
// RUN: %clang_cc1 -triple x86_64-pc-cygwin    -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP
// Darwin treats any array >= threshold as protectable even under basic -fstack-protector,
// so int[4] (16 bytes >= 8) acquires ssp.protected unlike on Linux:
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -stack-protector 1 | FileCheck %s --check-prefix=SSP,DARWIN
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -stack-protector 2 | FileCheck %s --check-prefix=STRONG

// Tests that Clang emits llvm.ssp.protected for local variables whose QualType contains a protectable array.
// Derived from llvm/test/CodeGen/X86/stack-protector.ll.

extern int printf(const char *, ...);
extern char *strcpy(char *, const char *);

// --- test1: large char array ---
// SSP-LABEL: @large_char_arr(
// SSP: call void @llvm.ssp.protected(ptr %buf)
// STRONG-LABEL: @large_char_arr(
// STRONG: call void @llvm.ssp.protected(ptr %buf)
void large_char_arr(const char *a) {
  char buf[16];
  strcpy(buf, a);
  printf("%s\n", buf);
}

// --- test2: struct containing large char array ---
// SSP-LABEL: @struct_large_char_arr(
// SSP: call void @llvm.ssp.protected(ptr %b)
// STRONG-LABEL: @struct_large_char_arr(
// STRONG: call void @llvm.ssp.protected(ptr %b)
struct foo { char buf[16]; };
void struct_large_char_arr(const char *a) {
  struct foo b;
  strcpy(b.buf, a);
  printf("%s\n", b.buf);
}

// --- test3: small char array ---
// SSP-LABEL: @small_char_arr(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @small_char_arr(
// STRONG: call void @llvm.ssp.protected(ptr %buf)
struct foo_small { char buf[4]; };
void small_char_arr(const char *a) {
  char buf[4];
  strcpy(buf, a);
  printf("%s\n", buf);
}

// --- test4: struct containing small char array ---
// SSP-LABEL: @struct_small_char_arr(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @struct_small_char_arr(
// STRONG: call void @llvm.ssp.protected(ptr %b)
void struct_small_char_arr(const char *a) {
  struct foo_small b;
  strcpy(b.buf, a);
  printf("%s\n", b.buf);
}

// --- test5: struct with scalar fields only (no arrays) ---
// SSP-LABEL: @struct_no_arr(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @struct_no_arr(
// STRONG-NOT: call void @llvm.ssp.protected
struct pair { int i; int j; };
void struct_no_arr(void) {
  struct pair p;
  p.i = 0; p.j = 1;
  printf("%d %d\n", p.i, p.j);
}

// --- test6: address of local scalar taken ---
// SSP-LABEL: @addr_of_local(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @addr_of_local(
// STRONG-NOT: call void @llvm.ssp.protected
void addr_of_local(void) {
  int a;
  int *j = &a;
  *j = 5;
  printf("%d\n", *j);
}

// --- test22: struct with tiny char array (mirrors "class A { char arr[2]; }") ---
// SSP-LABEL: @struct_tiny_char_arr(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @struct_tiny_char_arr(
// STRONG: call void @llvm.ssp.protected(ptr %a)
struct A { char arr[2]; };
signed char struct_tiny_char_arr(void) {
  struct A a;
  return a.arr[0];
}

// --- test23: char[2] nested in several layers of structs and unions ---
// SSP-LABEL: @nested_struct_tiny_arr(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @nested_struct_tiny_arr(
// STRONG: call void @llvm.ssp.protected(ptr %x)
struct inner23 { char arr[2]; };
union mid23 { struct inner23 s; };
struct outer23 { union mid23 u; };
signed char nested_struct_tiny_arr(void) {
  struct outer23 x;
  return x.u.s.arr[0];
}

// --- test25: integer array ---
// SSP-LABEL: @int_arr(
// DARWIN: call void @llvm.ssp.protected
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @int_arr(
// STRONG: call void @llvm.ssp.protected(ptr %a)
int int_arr(void) {
  int a[4];
  return a[0];
}

// --- test26: nested struct with no arrays ---
// SSP-LABEL: @nested_struct_no_arr(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @nested_struct_no_arr(
// STRONG-NOT: call void @llvm.ssp.protected
struct nest { struct pair left; struct pair right; };
void nested_struct_no_arr(void) {
  struct nest n;
  n.left.i = 0;
  printf("%d\n", n.left.i);
}

// --- test28: custom buffer-size = 33 ---
// SSP-LABEL: @custom33_not_protected(
// STRONG-LABEL: @custom33_not_protected(
// SSP33-LABEL: @custom33_not_protected(
// SSP33-NOT: call void @llvm.ssp.protected
//
// SSP-LABEL: @custom33_protected(
// STRONG-LABEL: @custom33_protected(
// SSP33-LABEL: @custom33_protected(
// SSP33: call void @llvm.ssp.protected(ptr %test)
int custom33_not_protected(void) { char test[32]; return test[0]; }
int custom33_protected(void)     { char test[33]; return test[0]; }

// --- test29: custom buffer-size = 5 ---
// SSP-LABEL: @custom5_not_protected(
// STRONG-LABEL: @custom5_not_protected(
// SSP5-LABEL: @custom5_not_protected(
// SSP5-NOT: call void @llvm.ssp.protected
//
// SSP-LABEL: @custom5_protected(
// STRONG-LABEL: @custom5_protected(
// SSP5-LABEL: @custom5_protected(
// SSP5: call void @llvm.ssp.protected(ptr %test)
int custom5_not_protected(void) { char test[4]; return test[0]; }
int custom5_protected(void)     { char test[5]; return test[0]; }

// --- test30: struct { int; char[5]; } with varying threshold ---
// SSP-LABEL: @struct_custom_arr(
// SSP-NOT: call void @llvm.ssp.protected
// STRONG-LABEL: @struct_custom_arr(
// STRONG: call void @llvm.ssp.protected(ptr %test)
// SSP5-LABEL: @struct_custom_arr(
// SSP5: call void @llvm.ssp.protected(ptr %test)
struct small_char { int i; char arr[5]; };
int struct_custom_arr(void) {
  struct small_char test;
  return test.arr[0];
}

// --- test31: struct containing large int array ---
// SSP-LABEL: @struct_large_int_arr(
// SSP-NOT: call void @llvm.ssp.protected(ptr %b)
// STRONG-LABEL: @struct_large_int_arr(
// STRONG: call void @llvm.ssp.protected(ptr %b)
struct foo_int { int buf[16]; };
int struct_large_int_arr() {
  struct foo_int b;
  return b.buf[0];
}