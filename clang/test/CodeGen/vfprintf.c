// RUN: %clang_cc1 -emit-llvm-only %s

typedef struct _IO_FILE FILE;
int vfprintf(FILE*restrict,const char*restrict, __builtin_va_list);
void foo(__builtin_va_list ap) {
  vfprintf(0, " ", ap);
}

