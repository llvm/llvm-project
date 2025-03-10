// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s        -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions -fcxx-exceptions    -fsized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s        -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions -fcxx-exceptions -fno-sized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s        -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions -fcxx-exceptions    -fsized-deallocation -fno-aligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s        -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions -fcxx-exceptions -fno-sized-deallocation -fno-aligned-allocation

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
}

using size_t = __SIZE_TYPE__;

struct Invalid1 {
  // expected-error@-1 {{declaration of type aware 'operator new' in 'Invalid1' must have matching type aware 'operator delete'}}
  void *operator new(std::type_identity<Invalid1>, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator new' declared here}}
};
struct Invalid2 {
  // expected-error@-1 {{declaration of type aware 'operator new' in 'Invalid2' must have matching type aware 'operator delete'}}
  template <class T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator new' declared here}}
};
struct Invalid3 {
  // expected-error@-1 {{declaration of type aware 'operator delete' in 'Invalid3' must have matching type aware 'operator new'}}
  void operator delete(std::type_identity<Invalid3>, void*, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator delete' declared here}}
};
struct Invalid4 {
  // expected-error@-1 {{declaration of type aware 'operator delete' in 'Invalid4' must have matching type aware 'operator new'}}
  template <class T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator delete' declared here}}
};
struct Invalid5 {
  // expected-error@-1 {{declaration of type aware 'operator new[]' in 'Invalid5' must have matching type aware 'operator delete[]'}}
  void *operator new[](std::type_identity<Invalid5>, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator new[]' declared here}}
};
struct Invalid6 {
  // expected-error@-1 {{declaration of type aware 'operator new[]' in 'Invalid6' must have matching type aware 'operator delete[]'}}
  template <class T> void *operator new[](std::type_identity<T>, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator new[]' declared here}}
};
struct Invalid7 {
  // expected-error@-1 {{declaration of type aware 'operator delete[]' in 'Invalid7' must have matching type aware 'operator new[]'}}
  void operator delete[](std::type_identity<Invalid7>, void*, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator delete[]' declared here}}
};
struct Invalid8 {
  // expected-error@-1 {{declaration of type aware 'operator delete[]' in 'Invalid8' must have matching type aware 'operator new[]'}}
  template <class T> void operator delete[](std::type_identity<T>, void*, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator delete[]' declared here}}
};

struct TestClass1 {
  void *operator new(std::type_identity<TestClass1>, size_t, std::align_val_t); // #TestClass1_new
  void  operator delete(std::type_identity<int>, void *, size_t, std::align_val_t); // #TestClass1_delete
};

struct TestClass2 {
  void *operator new(std::type_identity<int>, size_t, std::align_val_t); // #TestClass2_new
  void  operator delete(std::type_identity<TestClass2>, void *, size_t, std::align_val_t);  // #TestClass2_delete
};

void basic_tests() {
  TestClass1 * tc1 = new TestClass1;
  delete tc1;
  // expected-error@-1 {{no suitable member 'operator delete' in 'TestClass1'}}
  // expected-note@#TestClass1_delete {{member 'operator delete' declared here}}
  TestClass2 * tc2 = new TestClass2;
  // expected-error@-1 {{no matching function for call to 'operator new'}}
  delete tc2;
}

struct Baseclass1 {
  void *operator new(std::type_identity<Baseclass1>, size_t, std::align_val_t);
  void  operator delete(std::type_identity<Baseclass1>, void *, size_t, std::align_val_t); // #Baseclass1_delete
};

struct Subclass1 : Baseclass1 {
  Subclass1();
};

struct Baseclass2 {
  template <class T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  void  operator delete(std::type_identity<Baseclass2>, void *, size_t, std::align_val_t); // #Baseclass2_delete
};

struct Subclass2 : Baseclass2 {
  Subclass2();
};

struct Baseclass3 {
  template <class T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  template <class T> void  operator delete(std::type_identity<T>, void *, size_t, std::align_val_t);
};

struct Subclass3_1 : Baseclass3 {
  Subclass3_1();
  void *operator new(std::type_identity<int>, size_t, std::align_val_t);
  template <class T> void  operator delete(std::type_identity<T>, void *, size_t, std::align_val_t);
};

struct Subclass3_2 : Baseclass3 {
  Subclass3_2();
  template <class T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  void  operator delete(std::type_identity<int>, void *, size_t, std::align_val_t); // #Subclass3_2_delete
};


void test_subclasses() {
  Subclass1 * sc1 = new Subclass1;
  // expected-error@-1 {{no matching function for call to 'operator new'}}
  delete sc1;
  // expected-error@-1 {{no suitable member 'operator delete' in 'Subclass1'}}
  // expected-note@#Baseclass1_delete {{member 'operator delete' declared here}}
  Subclass2 * sc2 = new Subclass2;
  delete sc2;
  // expected-error@-1 {{no suitable member 'operator delete' in 'Subclass2'}}
  // expected-note@#Baseclass2_delete {{member 'operator delete' declared here}}
  Subclass3_1 * sc3_1 = new Subclass3_1;
  // expected-error@-1 {{no matching function for call to 'operator new'}}
  delete sc3_1;
  Subclass3_2 * sc3_2 = new Subclass3_2;
  delete sc3_2;
  // expected-error@-1 {{no suitable member 'operator delete' in 'Subclass3_2'}}
  // expected-note@#Subclass3_2_delete {{member 'operator delete' declared here}}
}
