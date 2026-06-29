// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify %s
// expected-no-diagnostics

// Test that __memory_scope enum is available without any includes

void test_basic_usage(void) {
  __memory_scope scope1 = __memory_scope_singlethread;
  __memory_scope scope2 = __memory_scope_wavefront;
  __memory_scope scope3 = __memory_scope_workgroup;
  __memory_scope scope4 = __memory_scope_cluster;
  __memory_scope scope5 = __memory_scope_device;
  __memory_scope scope6 = __memory_scope_system;
}

void test_assignment(void) {
  __memory_scope scope = __memory_scope_wavefront;
  scope = __memory_scope_device;
  scope = __memory_scope_system;
}

void test_comparison(void) {
  __memory_scope scope = __memory_scope_wavefront;

  if (scope == __memory_scope_wavefront) {
    // ok
  }

  if (scope != __memory_scope_device) {
    // ok
  }
}

void test_switch(__memory_scope scope) {
  switch (scope) {
    case __memory_scope_singlethread:
      break;
    case __memory_scope_wavefront:
      break;
    case __memory_scope_workgroup:
      break;
    case __memory_scope_cluster:
      break;
    case __memory_scope_device:
      break;
    case __memory_scope_system:
      break;
  }
}

void test_function_param(__memory_scope scope) {
  // ok
}

__memory_scope test_function_return(void) {
  return __memory_scope_device;
}
