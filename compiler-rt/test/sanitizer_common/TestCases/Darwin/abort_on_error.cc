// Check that sanitizers on OS X crash the process by default (i.e.
// abort_on_error=1). See also Linux/abort_on_error.cc.

// RUN: %clangxx %s -o %t

// Intentionally don't inherit the default options.
// RUN: env %tool_options='' not --crash %run %t 2>&1

// When we use lit's default options, we shouldn't crash.
// RUN: not %run %t 2>&1

int global;

int main() {
  volatile int *a = new int[100];
  delete[] a;
  global = a[0];  // use-after-free: triggers ASan report.
  return 0;
}
