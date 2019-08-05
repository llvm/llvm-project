// Check that sanitizers on OS X crash the process by default (i.e.
// abort_on_error=1). See also Linux/abort_on_error.cpp.

// RUN: %clangxx -DUSING_%tool_name %s -o %t

// Intentionally don't inherit the default options.
// RUN: env %tool_options='' not --crash %run %t 2>&1

// When we use lit's default options, we shouldn't crash.
// RUN: not %run %t 2>&1

// Leak detection isn't treated as an error so `abort_on_error=1` doesn't work.
// UNSUPPORTED: lsan

int global;

int main() {
#if defined(USING_ubsan)
  int value = 5;
  int computation = value / 0; // Division by zero.
#else
  volatile int *a = new int[100];
  delete[] a;
  global = a[0]; // use-after-free: triggers ASan/TSan report.
#endif
  return 0;
}
