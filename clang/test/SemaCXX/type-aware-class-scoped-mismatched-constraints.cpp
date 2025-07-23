// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s -std=c++26 -Wno-ext-cxx-type-aware-allocators -fexceptions -fcxx-exceptions    -fsized-deallocation    -faligned-allocation -Wno-non-c-typedef-for-linkage -DDEFAULT_DELETE=1
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s -std=c++26 -Wno-ext-cxx-type-aware-allocators -fexceptions -fcxx-exceptions -fno-sized-deallocation    -faligned-allocation -Wno-non-c-typedef-for-linkage -DDEFAULT_DELETE=0
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s -std=c++26 -Wno-ext-cxx-type-aware-allocators -fexceptions -fcxx-exceptions    -fsized-deallocation -fno-aligned-allocation -Wno-non-c-typedef-for-linkage -DDEFAULT_DELETE=1
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s -std=c++26 -Wno-ext-cxx-type-aware-allocators -fexceptions -fcxx-exceptions -fno-sized-deallocation -fno-aligned-allocation -Wno-non-c-typedef-for-linkage -DDEFAULT_DELETE=0

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
}

static_assert(__has_extension(cxx_type_aware_allocators), "Verifying the type aware extension flag is set");

using size_t = __SIZE_TYPE__;

void *operator new(size_t); // #default_operator_new

#if DEFAULT_DELETE==0
void operator delete(void*) noexcept; // #default_operator_delete
#elif DEFAULT_DELETE==1
void operator delete(void*, size_t) noexcept; // #default_operator_delete
#elif DEFAULT_DELETE==2
void operator delete(void*, std::align_val_t) noexcept; // #default_operator_delete
#elif DEFAULT_DELETE==3
void operator delete(void*, size_t, std::align_val_t) noexcept; // #default_operator_delete
#endif

struct Invalid1 {
  // expected-error@-1 {{declaration of type aware 'operator new' in 'Invalid1' must have matching type aware 'operator delete'}}
  void *operator new(std::type_identity<Invalid1>, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator new' declared here}}
};
struct Invalid2 {
  // expected-error@-1 {{declaration of type aware 'operator new' in 'Invalid2' must have matching type aware 'operator delete'}}
  template <class T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #Invalid2_new
  // expected-note@-1 {{unmatched type aware 'operator new' declared here}}
};
struct Invalid3 {
  // expected-error@-1 {{declaration of type aware 'operator delete' in 'Invalid3' must have matching type aware 'operator new'}}
  void operator delete(std::type_identity<Invalid3>, void*, size_t, std::align_val_t);
  // expected-note@-1 {{unmatched type aware 'operator delete' declared here}}
};
struct Invalid4 {
  // expected-error@-1 {{declaration of type aware 'operator delete' in 'Invalid4' must have matching type aware 'operator new'}}
  template <class T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #Invalid4_delete
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

// Invalid9 and Invalid10 will ensure we report the correct owner for the
// resolved, but unmatched, new and delete
struct Invalid9: Invalid2 {};
struct Invalid10: Invalid4 {};
// Invalid11 inherits a "matching" new and delete pair (so no inheritance ambiguity)
// but the resolved operators are from different scopes
struct Invalid11 : Invalid2, Invalid4 {};
struct Invalid12 : Invalid2, Invalid4 {
  using Invalid2::operator new;
  using Invalid4::operator delete;
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
  Invalid9 * i9 = new Invalid9;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#Invalid2_new {{type aware 'operator new' declared here in 'Invalid2'}}
  delete i9;
  Invalid10 * i10 = new Invalid10;
  // expected-error@-1 {{type aware 'operator delete' requires a matching type aware 'operator new' to be declared in the same scope}}
  // expected-note@#Invalid4_delete {{type aware 'operator delete' declared here in 'Invalid4'}}
  // expected-note@#default_operator_new {{non-type aware 'operator new' declared here in the global namespace}}
  delete i10;
  Invalid11 * i11 = new Invalid11;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#Invalid2_new {{type aware 'operator new' declared here in 'Invalid2'}}
  // expected-note@#Invalid4_delete {{type aware 'operator delete' declared here in 'Invalid4'}}
  delete i11;
  Invalid12 * i12 = new Invalid12;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#Invalid2_new {{type aware 'operator new' declared here in 'Invalid2'}}
  // expected-note@#Invalid4_delete {{type aware 'operator delete' declared here in 'Invalid4'}}
  delete i12;
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

template <class A, class B> constexpr bool same_type_v = false;
template <class A> constexpr bool same_type_v<A, A> = true;

template <class T> struct InvalidConstrainedOperator {
  template <class U> void *operator new(std::type_identity<U>, size_t, std::align_val_t);
  template <class U> requires (same_type_v<T, int>) void  operator delete(std::type_identity<U>, void *, size_t, std::align_val_t); // #InvalidConstrainedOperator_delete
};

struct Context;
template <class T> struct InvalidConstrainedCleanup {
  template <class U> void *operator new(std::type_identity<U>, size_t, std::align_val_t, Context&); // #InvalidConstrainedCleanup_placement_new
  template <class U> requires (same_type_v<T, int>) void operator delete(std::type_identity<U>, void *, size_t, std::align_val_t, Context&); // #InvalidConstrainedCleanup_delete
  template <class U> void operator delete(std::type_identity<U>, void *, size_t, std::align_val_t);
};

void test_incompatible_constrained_operators(Context &Ctx) {
  InvalidConstrainedOperator<int> *ico1 = new InvalidConstrainedOperator<int>;
  delete ico1;
  InvalidConstrainedOperator<float> *ico2 = new InvalidConstrainedOperator<float>;
  delete ico2;
  // expected-error@-1 {{no suitable member 'operator delete' in 'InvalidConstrainedOperator<float>'}}
  // expected-note@#InvalidConstrainedOperator_delete {{member 'operator delete' declared here}}
  InvalidConstrainedCleanup<int> *icc1 = new (Ctx) InvalidConstrainedCleanup<int>;
  delete icc1;
  InvalidConstrainedCleanup<float> *icc2 = new (Ctx) InvalidConstrainedCleanup<float>;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware placement 'operator delete' to be declared in the same scope}}
  // expected-note@#InvalidConstrainedCleanup_placement_new {{type aware 'operator new' declared here in 'InvalidConstrainedCleanup<float>'}}
  delete icc2;
}

typedef struct {
  // expected-error@-1 {{declaration of type aware 'operator new' in '(unnamed struct}}
  // expected-note@#AnonymousClass1_new {{unmatched type aware 'operator new' declared here}}
  template <class T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #AnonymousClass1_new
} AnonymousClass1;

typedef struct {
  // expected-error@-1 {{declaration of type aware 'operator delete' in '(unnamed struct}}
  template <class T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #AnonymousClass2_delete
} AnonymousClass2;

typedef struct {
  template <class T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  template <class T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t);
} AnonymousClass3;

using AnonymousClass4 = struct {};
using AnonymousClass5 = struct {};
using AnonymousClass6 = struct {};
using AnonymousClass7 = struct {
  // expected-error@-1 {{declaration of type aware 'operator new' in}}
  // expected-note@#AnonymousClass7_new {{unmatched type aware 'operator new' declared here}}
  template <class T> void *operator new(std::type_identity<T>, size_t, std::align_val_t, Context&); // #AnonymousClass7_new
};


void *operator new(std::type_identity<AnonymousClass4>, size_t, std::align_val_t); // #AnonymousClass4_new
void operator delete(std::type_identity<AnonymousClass5>, void*, size_t, std::align_val_t); // #AnonymousClass5_delete
void *operator new(std::type_identity<AnonymousClass6>, size_t, std::align_val_t, Context&); // #AnonymousClass6_placement_new
void operator delete(std::type_identity<AnonymousClass6>, void*, size_t, std::align_val_t);
void test_anonymous_types(Context &Ctx) {
  AnonymousClass1 *ac1 = new AnonymousClass1;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#AnonymousClass1_new {{type aware 'operator new' declared here in 'AnonymousClass1'}}

  delete ac1;
  AnonymousClass2 *ac2 = new AnonymousClass2;
  // expected-error@-1 {{type aware 'operator delete' requires a matching type aware 'operator new' to be declared in the same scope}}
  // expected-note@#AnonymousClass2_delete {{unmatched type aware 'operator delete' declared here}}
  // expected-note@#AnonymousClass2_delete {{type aware 'operator delete' declared here in 'AnonymousClass2'}}
  // expected-note@#default_operator_new {{non-type aware 'operator new' declared here in the global namespace}}

  delete ac2;
  AnonymousClass3 *ac3 = new AnonymousClass3;
  delete ac3;
  AnonymousClass4 *ac4 = new AnonymousClass4;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#AnonymousClass4_new {{type aware 'operator new' declared here in the global namespace}}
  delete ac4;
  AnonymousClass5 *ac5 = new AnonymousClass5;
  // expected-error@-1 {{type aware 'operator delete' requires a matching type aware 'operator new' to be declared in the same scope}}
  // expected-note@#AnonymousClass5_delete {{type aware 'operator delete' declared here}}
  // expected-note@#default_operator_new {{non-type aware 'operator new' declared here in the global namespace}}
  delete ac5;
  AnonymousClass6 *ac6 = new (Ctx) AnonymousClass6;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware placement 'operator delete' to be declared in the same scope}}
  // expected-note@#AnonymousClass6_placement_new {{type aware 'operator new' declared here in the global namespace}}
  // expected-note@#default_operator_delete {{non-type aware 'operator delete' declared here in the global namespace}}
  delete ac6;
}