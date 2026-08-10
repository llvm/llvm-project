// RUN: %check_clang_tidy %s performance-no-int-to-ptr %t

// can't implicitly cast int to a pointer.
// can't use static_cast<>() to cast integer to a pointer.
// can't use dynamic_cast<>() to cast integer to a pointer.
// can't use const_cast<>() to cast integer to a pointer.

void *t0(long long int x) {
  return (void *)x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}

void *t1(int x) {
  return reinterpret_cast<void *>(x);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}

// Don't diagnose casts from integer literals.
// It's a widely-used technique in embedded/microcontroller/hardware interfacing.
void *t3(long long int x) {
  return (void *)0xFEEDFACE;
}
