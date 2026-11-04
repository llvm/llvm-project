// RUN: %clang_cc1 -triple x86_64-linux-gnu -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OPT --implicit-check-not=strncmp.inline
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,O0 --implicit-check-not=strncmp.inline
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O3 -disable-llvm-passes -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OPT --implicit-check-not=strncmp.inline
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=gnu89 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OPT --implicit-check-not=strncmp.inline
//
// GH110774: a GNU89 extern inline redefinition of an inline builtin (an
// always_inline gnu_inline definition of a library builtin) must not emit a
// second body into the ".inline" clone.

typedef unsigned long size_t;
#define AVAILABLE_EXTERNALLY extern inline __attribute__((always_inline)) __attribute__((gnu_inline))

// CHECK: @pstrchr ={{.*}} global ptr @strchr

// The reproducer: the first definition is emitted, the second is skipped.
AVAILABLE_EXTERNALLY int memcmp(const void *a, const void *b, size_t c) {
  return __builtin_memcmp(a, b, c);
}

int bar(const void *a, const void *b, size_t c) {
  return memcmp(a, b, c);
}

AVAILABLE_EXTERNALLY int memcmp(const void *a, const void *b, size_t c) {
  return __builtin_memcmp(a, b, c);
}

// CHECK-LABEL: define{{.*}} i32 @bar(
// CHECK:         call i32 @memcmp.inline(
// CHECK:       declare{{.*}} i32 @memcmp(
// CHECK-LABEL: define internal i32 @memcmp.inline(
// CHECK:         call i32 @memcmp(

// Three definitions.
AVAILABLE_EXTERNALLY void *memcpy(void *d, const void *s, size_t n) {
  return __builtin_memcpy(d, s, n);
}

void *copy(void *d, const void *s, size_t n) {
  return memcpy(d, s, n);
}

AVAILABLE_EXTERNALLY void *memcpy(void *d, const void *s, size_t n) {
  return __builtin_memcpy(d, s, n);
}

AVAILABLE_EXTERNALLY void *memcpy(void *d, const void *s, size_t n) {
  return __builtin_memcpy(d, s, n);
}

// CHECK-LABEL: define{{.*}} ptr @copy(
// CHECK:         call ptr @memcpy.inline(
// CHECK:       declare{{.*}} ptr @memcpy(
// CHECK-LABEL: define internal ptr @memcpy.inline(
// CHECK:         call void @llvm.memcpy.p0.p0.i64(

// Address taken between the definitions: no clone exists before emission.
AVAILABLE_EXTERNALLY char *strchr(const char *s, int c) {
  return __builtin_strchr(s, c);
}

char *(*pstrchr)(const char *, int) = strchr;

AVAILABLE_EXTERNALLY char *strchr(const char *s, int c) {
  return __builtin_strchr(s, c);
}

// CHECK:       declare{{.*}} ptr @strchr(

// Extern inline followed by a strong definition: the strong one wins.
AVAILABLE_EXTERNALLY int strncmp(const char *a, const char *b, size_t n) {
  return __builtin_strncmp(a, b, n);
}

int cmp1(const char *a, const char *b, size_t n) {
  return strncmp(a, b, n);
}

int strncmp(const char *a, const char *b, size_t n) {
  return 0;
}

int cmp2(const char *a, const char *b, size_t n) {
  return strncmp(a, b, n);
}

// CHECK-LABEL: define{{.*}} i32 @cmp1(
// CHECK:         call i32 @strncmp(
// CHECK-LABEL: define{{.*}} i32 @strncmp(
// CHECK-LABEL: define{{.*}} i32 @cmp2(
// CHECK:         call i32 @strncmp(

// Not an inline builtin without always_inline: ordinary available_externally.
extern inline __attribute__((gnu_inline)) size_t strlen(const char *s) {
  return *s ? 1 : 0;
}

size_t len(const char *s) {
  return strlen(s);
}

extern inline __attribute__((gnu_inline)) size_t strlen(const char *s) {
  return *s ? 1 : 0;
}

// CHECK-LABEL: define{{.*}} i64 @len(
// CHECK:         call i64 @strlen(
// OPT:          define available_externally i64 @strlen(
// O0:           declare{{.*}} i64 @strlen(

// Control: not a library builtin.
AVAILABLE_EXTERNALLY int foo(int x) {
  return x + 1;
}

int use_foo(int x) {
  return foo(x);
}

AVAILABLE_EXTERNALLY int foo(int x) {
  return x + 1;
}

// CHECK-LABEL: define{{.*}} i32 @use_foo(
// CHECK:         call i32 @foo(
// CHECK-LABEL: define available_externally i32 @foo(

// CHECK-LABEL: define internal ptr @strchr.inline(
