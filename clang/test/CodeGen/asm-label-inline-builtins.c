// RUN: %clang_cc1 -triple x86_64 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// Verifies that clang-generated *.inline carry the same name at call and callee
// site, in spite of asm labels.

typedef struct _IO_FILE FILE;
extern FILE *stdout;
extern int vprintf (const char *__restrict __format, __builtin_va_list __arg);
extern int __vfprintf_chk (FILE *__restrict __stream, int __flag,
      const char *__restrict __format, __builtin_va_list __ap);
extern int __vprintf_chk (int __flag, const char *__restrict __format,
     __builtin_va_list __ap);

extern __typeof (vprintf) vprintf __asm ("__vprintfieee128");
extern __typeof (__vfprintf_chk) __vfprintf_chk __asm ("__vfprintf_chkieee128");
extern __typeof (__vprintf_chk) __vprintf_chk __asm ("__vprintf_chkieee128");

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
vprintf (const char *__restrict __fmt, __builtin_va_list __ap)
{
  return __vfprintf_chk (stdout, 2 - 1, __fmt, __ap);
}

void test(const char *fmt, __builtin_va_list ap) {
  vprintf(fmt, ap);
}

// CHECK-LABEL: void @test(
// CHECK: call i32 @__vprintfieee128.inline(
//
// CHECK-LABEL: internal i32 @__vprintfieee128.inline(
// CHECK: call i32 @__vfprintf_chkieee128(
