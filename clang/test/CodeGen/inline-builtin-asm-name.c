// RUN: %clang_cc1 -triple i686-windows-gnu -emit-llvm -o - %s -disable-llvm-optzns | FileCheck %s

// CHECK: call i32 @"\01_asm_func_name.inline"

// CHECK: declare dso_local i32 @"\01_asm_func_name"(ptr noundef, i32 noundef, ptr noundef, ptr noundef)

// CHECK: define internal i32 @"\01_asm_func_name.inline"

// CHECK: call i32 @__mingw_vsnprintf

// CHECK: declare dso_local i32 @__mingw_vsnprintf

typedef unsigned int size_t;

int __mingw_vsnprintf(char *_DstBuf, size_t _MaxCount, const char *_Format, __builtin_va_list _ArgList);

// For the real use case, "_asm_func_name" is actually "___mingw_vsnprintf", but it's renamed in the testcase for disambiguation.
int vsnprintf(char *__stream, size_t __n, const char *__format, __builtin_va_list __local_argv) __asm__("_asm_func_name");

extern __inline__ __attribute__((__always_inline__, __gnu_inline__))
int vsnprintf(char *__stream, size_t __n, const char *__format, __builtin_va_list __local_argv)
{
  return __mingw_vsnprintf(__stream, __n, __format, __local_argv);
}

void call(const char* fmt, ...) {
  char buf[200];
  __builtin_va_list ap;
  __builtin_va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  __builtin_va_end(ap);
}
