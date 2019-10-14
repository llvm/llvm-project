// REQUIRES: static-analyzer
// RUN: clang-tidy %s -checks='-*,clang-analyzer-unix.Malloc' -config='{CheckOptions: [{ key: "clang-analyzer-unix.DynamicMemoryModeling:Optimistic", value: true}]}' -- | FileCheck %s
typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void __attribute((ownership_returns(malloc))) *my_malloc(size_t);
void __attribute((ownership_takes(malloc, 1))) my_free(void *);

void f1() {
  void *p = malloc(12);
  return;
  // CHECK: warning: Potential leak of memory pointed to by 'p' [clang-analyzer-unix.Malloc]
}

void af2() {
  void *p = my_malloc(12);
  my_free(p);
  free(p);
  // CHECK: warning: Attempt to free released memory [clang-analyzer-unix.Malloc]
}
