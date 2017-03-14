// RUN: %clang_analyze_cc1 -triple gcc-linaro-arm-linux-gnueabihf -analyzer-checker=core,valist.Uninitialized,valist.CopyToSelf -analyzer-output=text -analyzer-store=region -verify %s
// expected-no-diagnostics

typedef unsigned int size_t;
typedef __builtin_va_list __gnuc_va_list;
typedef __gnuc_va_list va_list;

extern int vsprintf(char *__restrict __s,
                    const char *__restrict __format, __gnuc_va_list
                                                         __arg);

void _dprintf(const char *function, int flen, int line, int level,
             const char *prefix, const char *fmt, ...) {
  char raw[10];
  int err;
  va_list ap;

  __builtin_va_start(ap, fmt);
  err = vsprintf(raw, fmt, ap);
  __builtin_va_end(ap);
}
