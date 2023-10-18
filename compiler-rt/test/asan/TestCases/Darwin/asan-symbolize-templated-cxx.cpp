// UNSUPPORTED: ios
// RUN: %clangxx_asan -O0 -g %s -o %t.executable
// RUN: %env_asan_opts="symbolize=0" not %run %t.executable > %t_no_module_map.log 2>&1
// RUN: %asan_symbolize --force-system-symbolizer < %t_no_module_map.log 2>&1 | FileCheck %s
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>

// This test is deliberately convoluted so that there is a function call
// in the stack trace that contains nested parentheses.

template <class CallBackTy>
class IntWrapper {
  int value_;
  std::function<CallBackTy> callback_;

public:
  IntWrapper(int value, std::function<CallBackTy> callback) : value_(value), callback_(callback) {}
  int &operator=(const int &new_value) {
    value_ = new_value;
    callback_(value_);
  }
};

using IntW = IntWrapper<void(int)>;
IntW *a;

template <class T>
void writeToA(T new_value) {
  // CHECK: heap-use-after-free bar
  // NOTE: atos seems to emit the `void` return type here for some reason.
  // CHECK: #{{[0-9]+}} 0x{{.+}} in {{(void +)?}}writeToA<IntWrapper<void{{ *}}(int)>{{ *}}>(IntWrapper<void{{ *}}(int)>) asan-symbolize-templated-cxx.cpp:[[@LINE+1]]
  *a = new_value;
}

extern "C" void callback(int new_value) {
  printf("new value is %d\n", new_value);
}

template <class T, class V>
struct Foo {
  std::function<T> call;
  Foo(std::function<T> c) : call(c) {}
  void doCall(V new_value) {
    // CHECK: #{{[0-9]+}} 0x{{.+}} in Foo<void (IntWrapper<void{{ *}}(int)>),{{ *}}IntWrapper<void{{ *}}(int)>{{ *}}>::doCall(IntWrapper<void{{ *}}(int)>) asan-symbolize-templated-cxx.cpp:[[@LINE+1]]
    call(new_value);
  }
};

int main() {
  a = new IntW(0, callback);
  assert(a);
  // Foo<void(IntWrapper<void(int)>)>
  // This type is deliberately convoluted so that the demangled type contains nested parentheses.
  // In particular trying to match parentheses using a least-greedy regex approach will fail.
  Foo<void(IntW), IntW> foo(writeToA<IntW>);
  delete a;
  // CHECK: #{{[0-9]+}} 0x{{.+}} in main asan-symbolize-templated-cxx.cpp:[[@LINE+1]]
  foo.doCall(IntW(5, callback)); // BOOM
  return 0;
}
