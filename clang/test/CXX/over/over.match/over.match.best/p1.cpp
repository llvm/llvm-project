// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T> int &f0(T*, int);
float &f0(void*, int);

void test_f0(int* ip, void *vp) {
  // One argument is better...
  int &ir = f0(ip, 0);
  
  // Prefer non-templates to templates
  float &fr = f0(vp, 0);
}

// Partial ordering of function template specializations will be tested 
// elsewhere
// FIXME: Initialization by user-defined conversion is tested elsewhere
