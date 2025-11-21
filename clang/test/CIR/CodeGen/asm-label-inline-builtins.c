// RUN: %clang_cc1 -triple x86_64 -fclangir -emit-cir -disable-llvm-passes -o %t-cir.cir %s
// RUN: FileCheck --input-file=%t-cir.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64 -fclangir -emit-llvm -disable-llvm-passes -o %t-cir.ll %s
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -disable-llvm-passes -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG


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

// CIR: cir.func internal private @__vprintfieee128.inline({{.*}}) -> !s32i inline(always)
// CIR:   cir.call @__vfprintf_chkieee128(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
//
// CIR: cir.func {{.*}} @test({{.*}})
// CIR:   cir.call @__vprintfieee128.inline(%{{.*}}, %{{.*}})


// LLVM: define internal i32 @__vprintfieee128.inline({{.*}}) #[[ALWAYS_INLINE_ATTR:.*]] {
// LLVM:   call i32 @__vfprintf_chkieee128(ptr %{{.*}}, i32 1, ptr %{{.*}}, ptr %{{.*}})
//
// LLVM: define {{.*}} void @test{{.*}}
// LLVM:   call i32 @__vprintfieee128.inline(ptr %{{.*}}, ptr %{{.*}})
//
// LLVM: attributes #[[ALWAYS_INLINE_ATTR]] = { alwaysinline }

// Note: OGCG emits these in the opposite order, but the content is the same.


// OGCG: define {{.*}} void @test{{.*}}
// OGCG:   call i32 @__vprintfieee128.inline(ptr noundef %{{.*}}, ptr noundef %{{.*}})
//
// OGCG: define internal i32 @__vprintfieee128.inline({{.*}}) #[[ALWAYS_INLINE_ATTR:.*]] {
// OGCG:   call i32 @__vfprintf_chkieee128(ptr noundef %{{.*}}, i32 noundef 1, ptr noundef %{{.*}}, ptr noundef %{{.*}})
//
// OGCG: attributes #[[ALWAYS_INLINE_ATTR]] = { alwaysinline {{.*}} }
