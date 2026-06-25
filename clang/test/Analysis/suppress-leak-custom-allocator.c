// RUN: %clang_analyze_cc1 -analyzer-checker=unix.Malloc,unix.DynamicMemoryModeling \
// RUN: -analyzer-config unix.DynamicMemoryModeling:SuppressLeakReportsFor=arena \
// RUN: %s -verify

void *arena_alloc(void) __attribute__((ownership_returns(arena)));

void test() {
  void *p = arena_alloc();
  // expected-no-diagnostics
}