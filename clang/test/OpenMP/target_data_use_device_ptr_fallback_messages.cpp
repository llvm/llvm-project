// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=60 -verify=omp60,expected -ferror-limit 200 %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=61 -verify=omp61,expected -ferror-limit 200 %s

void f1(int x, int *p, int *q) {

  // Test that fallback modifier is only recognized in OpenMP 6.1+
#pragma omp target data map(x) use_device_ptr(fb_preserve: p) // omp60-error {{use of undeclared identifier 'fb_preserve'}}
  {}

#pragma omp target data map(x) use_device_ptr(fb_nullify: p) // omp60-error {{use of undeclared identifier 'fb_nullify'}}
  {}

  // Without modifier (should work in both versions)
#pragma omp target data map(x) use_device_ptr(p)
  {}

  // Unknown modifier: should fail in both versions
#pragma omp target data map(x) use_device_ptr(fb_abc: p) // expected-error {{use of undeclared identifier 'fb_abc'}}
  {}

  // Multiple modifiers: should fail in both versions
#pragma omp target data map(x) use_device_ptr(fb_nullify, fb_preserve: p, q) // omp61-error {{missing ':' after fallback modifier}} omp61-error {{expected expression}} omp61-error {{use of undeclared identifier 'fb_preserve'}} omp60-error {{use of undeclared identifier 'fb_nullify'}} omp60-error {{use of undeclared identifier 'fb_preserve'}}
  {}

  // Interspersed modifiers/list-items: should fail in both versions
#pragma omp target data map(x) use_device_ptr(fb_nullify: p, fb_preserve: q) // omp61-error {{use of undeclared identifier 'fb_preserve'}} omp60-error {{use of undeclared identifier 'fb_nullify'}} omp60-error {{use of undeclared identifier 'fb_preserve'}}
  {}

  // Test missing colon after modifier in OpenMP 6.1 - should error
#pragma omp target data map(x) use_device_ptr(fb_preserve p) // omp61-error {{missing ':' after fallback modifier}} omp60-error {{use of undeclared identifier 'fb_preserve'}}
  {}
}
