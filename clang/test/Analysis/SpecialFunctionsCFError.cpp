// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.coreFoundation.CFError \
// RUN:   -verify %s

typedef unsigned long size_t;
struct __CFError {};
typedef struct __CFError *CFErrorRef;
void *malloc(size_t);

class Foo {
public:
  Foo(CFErrorRef *error) {} // no-warning

  void operator delete(void *pointer, CFErrorRef *error) { // no-warning
    return;
  }

  void operator delete[](void *pointer, CFErrorRef *error) { // no-warning
    return;
  }

  // Check that we report warnings for operators when it can be useful
  void operator()(CFErrorRef *error) {} // expected-warning {{Function accepting CFErrorRef* should have a non-void return value to indicate whether or not an error occurred}}
};

// Check that global delete operator is not bothered as well
void operator delete(void *pointer, CFErrorRef *error) { // no-warning
  return;
}
