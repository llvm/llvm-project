// Check that sanitizers call _exit() on Linux by default (i.e.
// abort_on_error=0). See also Darwin/abort_on_error.cpp.

// RUN: %clangxx %s -o %t

// Intentionally don't inherit the default options.
// RUN: env %tool_options='' not %run %t 2>&1

// When we use lit's default options, we shouldn't crash either. On Linux
// lit doesn't set options anyway.
// RUN: not %run %t 2>&1

// Android needs abort_on_error=0
// UNSUPPORTED: android

namespace __sanitizer {
void Die();
}

int main() {
  __sanitizer::Die();
  return 0;
}
