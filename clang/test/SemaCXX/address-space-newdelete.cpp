// RUN: %clang_cc1 -fsyntax-only -verify %s

void* operator new (__SIZE_TYPE__ size, void* ptr);
void* operator new[](__SIZE_TYPE__ size, void* ptr);

typedef int __attribute__((address_space(1))) int_1;

void test_new(void *p) {
  (void)new int_1; // expected-error{{'new' cannot allocate objects of type 'int' in address space '1'}}
  (void)new __attribute__((address_space(1))) int; // expected-error{{'new' cannot allocate objects of type 'int' in address space '1'}}
  (void)new int_1 [5]; // expected-error{{'new' cannot allocate objects of type 'int' in address space '1'}}
  (void)new __attribute__((address_space(1))) int [5]; // expected-error{{'new' cannot allocate objects of type 'int' in address space '1'}}

  // Placement new
  (void)new (p) int_1; // expected-error{{'new' cannot allocate objects of type 'int' in address space '1'}}
  (void)new (p) __attribute__((address_space(1))) int; // expected-error{{'new' cannot allocate objects of type 'int' in address space '1'}}
  (void)new (p) int_1 [5]; // expected-error{{'new' cannot allocate objects of type 'int' in address space '1'}}
  (void)new (p) __attribute__((address_space(1))) int [5]; // expected-error{{'new' cannot allocate objects of type 'int' in address space '1'}}
}

void test_delete(int_1 *ip1) {
  delete ip1; // expected-error{{'delete' cannot delete objects of type 'int' in address space '1'}}
  delete [] ip1; // expected-error{{'delete' cannot delete objects of type 'int' in address space '1'}}
}
