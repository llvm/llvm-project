// Test that function name is mangled in the "created by an allocation" line,
// and demangled in the single-frame "stack trace" that follows.

// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s < %t.out

__attribute__((noinline))
int f() {
  int x;
  int *volatile p = &x;
  return *p;
}

int main(int argc, char **argv) {
  return f();
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: Uninitialized value was created by an allocation of 'x' in the stack frame
  // CHECK: #0 {{.*}} in f{{.*}} {{.*}}report-demangling.cpp:[[@LINE-9]]
}
