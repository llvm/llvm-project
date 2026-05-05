// RUN: %clang_cc1 -ftyped-memory-operations -fsyntax-only -verify %s
#define _TYPED_ALLOC(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *malloc(unsigned long);
void *typed_malloc(unsigned long, unsigned long long);
// expected-note@-1 {{rewrite target here}}
void *typed_malloc2(unsigned long, unsigned long long);
void *calloc(unsigned long, unsigned long);
void *typed_calloc(unsigned long, unsigned long, unsigned long long);
// expected-note@-1 2 {{rewrite target here}}
void *typed_calloc2(unsigned long, unsigned long, unsigned long long);

// Some function defs that don't match the required interface
void *invalid_typed_malloc1();
// expected-note@-1 {{rewrite target here}}
void *invalid_typed_malloc2(double);
// expected-note@-1 {{rewrite target here}}
int invalid_typed_malloc3(unsigned);
// expected-note@-1 3 {{rewrite target here}}
int invalid_typed_malloc4(__SIZE_TYPE__, unsigned long long);
void* invalid_typed_malloc5(__SIZE_TYPE__, int);

struct Foo {
  static void *malloc(unsigned long);
  // expected-note@-1 {{rewrite target here}}
  template <typename T> static void *template_malloc(__SIZE_TYPE__, unsigned long long);
  // expected-note@-1 {{candidate function template}}
  template <typename T> static void *test_malloc1(T) _TYPED_ALLOC(typed_malloc, 1);
  static void *typed_malloc_method(unsigned, unsigned long long);
  void *invalid_typed_malloc_method(unsigned, unsigned long long);
  void *method_malloc(unsigned) _TYPED_ALLOC(typed_malloc_method, 1);
  // expected-error@-1 {{typed memory operation 'method_malloc' cannot be an instance method}}
  void *method_malloc2(unsigned) _TYPED_ALLOC(invalid_typed_malloc_method, 1);
  // expected-error@-1 {{call to non-static member function without an object argument}}
  static void *class_typed_malloc(__SIZE_TYPE__, __SIZE_TYPE__);
  static void *class_malloc(__SIZE_TYPE__) _TYPED_ALLOC(class_typed_malloc, 1);
};

template <typename T> struct Bar {
  static void *class_typed_malloc(__SIZE_TYPE__, T);
  static void *class_malloc(__SIZE_TYPE__) _TYPED_ALLOC(class_typed_malloc, 1);
};

void *my_malloc2(int size) _TYPED_ALLOC(typed_malloc, 1);
// expected-error@-1 {{typed memory operation 'typed_malloc' has incompatible type ('void *(int, unsigned long)' vs 'void *(unsigned long, unsigned long long)')}}
void *my_malloc3(double size) _TYPED_ALLOC(typed_malloc, 1);
// expected-error@-1 {{invalid parameter type for inference at index 1. 'double' is not an integer type}}
void *my_malloc4(unsigned size) _TYPED_ALLOC(typed_malloc, -1);
// expected-error@-1 {{'typed_memory_operation' attribute parameter 1 is out of bounds}}
void *my_malloc4(unsigned size) _TYPED_ALLOC(typed_malloc, 0);
// expected-error@-1 {{'typed_memory_operation' attribute parameter 1 is out of bounds}}
void *my_malloc6(unsigned size) _TYPED_ALLOC(typed_malloc, 2);
// expected-error@-1 {{'typed_memory_operation' attribute parameter 1 is out of bounds}}

void *my_malloc_invalid1(unsigned size) _TYPED_ALLOC(invalid_typed_malloc0, 1);
// expected-error@-1 {{use of undeclared identifier 'invalid_typed_malloc0'}}
void *my_malloc_invalid2(unsigned size) _TYPED_ALLOC(invalid_typed_malloc1, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc1' has incompatible type ('void *(unsigned int, unsigned long)' vs 'void *()')}}
void *my_malloc_invalid3(unsigned size) _TYPED_ALLOC(invalid_typed_malloc2, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc2' has incompatible type ('void *(unsigned int, unsigned long)' vs 'void *(double)')}}
void *my_malloc_invalid4(unsigned size) _TYPED_ALLOC(invalid_typed_malloc3, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc3' has incompatible type ('void *(unsigned int, unsigned long)' vs 'int (unsigned int)')}}
// intentionally using the wrong function
void *my_malloc_invalid5(unsigned size) _TYPED_ALLOC(invalid_typed_malloc3, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc3' has incompatible type ('void *(unsigned int, unsigned long)' vs 'int (unsigned int)')}}
void *my_malloc_invalid6(unsigned size) _TYPED_ALLOC(invalid_typed_malloc3, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc3' has incompatible type ('void *(unsigned int, unsigned long)' vs 'int (unsigned int)')}}

void *my_malloc12(unsigned long size) _TYPED_ALLOC(calloc, 1);
void *my_malloc13(unsigned size) _TYPED_ALLOC(typed_calloc, 1);
// expected-error@-1 {{typed memory operation 'typed_calloc' has incompatible type ('void *(unsigned int, unsigned long)' vs 'void *(unsigned long, unsigned long, unsigned long long)')}}

void *my_malloc14(unsigned size) _TYPED_ALLOC(Foo::malloc, 1);
// expected-error@-1 {{typed memory operation 'malloc' has incompatible type ('void *(unsigned int, unsigned long)' vs 'void *(unsigned long)')}}
void *my_malloc15(unsigned size) _TYPED_ALLOC(Foo::template_malloc, 1);
// expected-error@-1 {{typed memory operation 'template_malloc' has multiple overloads}}
void *my_malloc16(__SIZE_TYPE__ size) _TYPED_ALLOC(Foo::template_malloc<unsigned>, 1);
void *my_malloc17(__SIZE_TYPE__ size) _TYPED_ALLOC(Foo::template_malloc<double>, 1);

void *my_calloc1(unsigned count, unsigned size) _TYPED_ALLOC(typed_calloc, 2);
// expected-error@-1 {{typed memory operation 'typed_calloc' has incompatible type ('void *(unsigned int, unsigned int, unsigned long)' vs 'void *(unsigned long, unsigned long, unsigned long long)')}}

// Overloading vs. rewrite

void alloc_overload_target1(__SIZE_TYPE__, __SIZE_TYPE__, float);
void alloc_overload_target2(__SIZE_TYPE__, __SIZE_TYPE__, int);
void alloc_overload(__SIZE_TYPE__, float) _TYPED_ALLOC(alloc_overload_target1, 1);
void alloc_overload(__SIZE_TYPE__, int) _TYPED_ALLOC(alloc_overload_target2, 1);

// Redeclarations
void *typed_for_redecl(unsigned long long, unsigned long long, unsigned long long);
void *typed_for_redecl2(unsigned long long, unsigned long long, unsigned long long);
void *my_calloc2(unsigned long long count, unsigned long long size) _TYPED_ALLOC(typed_for_redecl, 2);
void *my_calloc2(unsigned long long count, unsigned long long size) _TYPED_ALLOC(typed_for_redecl, 2);
// expected-note@-1 2 {{conflicting attribute is here}}
void *my_calloc2(unsigned long long count, unsigned long long size) _TYPED_ALLOC(typed_for_redecl, 1);
// expected-error@-1 {{attribute 'typed_memory_operation' is already applied with different arguments}}
void *my_calloc2(unsigned long long count, unsigned long long size) _TYPED_ALLOC(typed_for_redecl2, 2);
// expected-error@-1 {{attribute 'typed_memory_operation' is already applied with different arguments}}

void f(){
  my_malloc17(1);
  Bar<float>::class_malloc(10);
  Bar<__SIZE_TYPE__>::class_malloc(10);
  
}
