// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_pgogen -ffunction-sections main.c -c -o main.o
// RUN: %clang_pgogen -ffunction-sections needs_gc.c -c -o needs_gc.o
// RUN: %clang_pgogen main.o needs_gc.o -o needs_gc.out
// RUN: env LLVM_PROFILE_FILE=needs_gc.profraw %run ./needs_gc.out > /dev/null
// RUN: llvm-profdata show --all-functions needs_gc.profraw | FileCheck %s

// CHECK-DAG: main
// CHECK-DAG: baz
// CHECK-DAG: get_message

//--- main.c
const char *get_message(void) { return "Hello World!"; }

const char *baz();

int printf(const char *, ...);

int main(void) { printf("%s\n", baz()); }

//--- needs_gc.c
extern int not_def_one(const char *);
extern double not_def_two(void);

extern const char *get_message(void);

char buf[512];
int foo(const char *ptr, unsigned long size) {
  void *memcpy(void *, const void *, unsigned long);
  memcpy(buf, ptr, size);
  return not_def_one(buf);
}

double bar(void) { return not_def_two(); }

const char *baz() { return get_message(); }
