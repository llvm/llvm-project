// RUN: %clang_cc1 -std=c++1z -fno-sized-deallocation -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++03 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++03 -faligned-allocation -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -fsized-deallocation %s

#if !__has_builtin(__builtin_operator_new) || !__has_builtin(__builtin_operator_delete)
#error builtins should always be available
#endif

#if __has_builtin(__builtin_operator_new) != 201802L || \
    __has_builtin(__builtin_operator_delete) != 201802L
#error builtin should report updated value
#endif

typedef __SIZE_TYPE__ size_t;
namespace std {
  struct nothrow_t {};
#if __cplusplus >= 201103L
enum class align_val_t : size_t {};
#else
  enum align_val_t {
  // We can't force an underlying type when targeting windows.
# ifndef _WIN32
    __zero = 0, __max = (size_t)-1
# endif
  };
#endif
}
std::nothrow_t nothrow;

void *operator new(size_t); // expected-note 1+ {{candidate function}}
void operator delete(void *); // expected-note 1+ {{candidate function}}

// Declare the reserved placement operators.
void *operator new(size_t, void*) throw(); // expected-note 1+ {{candidate function}}
void operator delete(void *, void *)throw(); // expected-note 1+ {{candidate function}}
void *operator new[](size_t, void*) throw();
void operator delete[](void*, void*) throw();

// Declare the replaceable global allocation operators.
void *operator new(size_t, const std::nothrow_t &) throw(); // expected-note 1+ {{candidate function}}
void *operator new[](size_t, const std::nothrow_t &) throw();
void operator delete(void *, const std::nothrow_t &)throw(); // expected-note 1+ {{candidate function}}
void operator delete[](void *, const std::nothrow_t &) throw();

// aligned allocation and deallocation functions.
void* operator new  ( size_t count, std::align_val_t al); // expected-note 1+ {{candidate function}}
void operator delete(void *, std::align_val_t); // expected-note 1+ {{candidate}}
#ifndef __cpp_aligned_new
// expected-note@-3 1+ {{non-usual 'operator new' declared here}}
// expected-note@-3 1+ {{non-usual 'operator delete' declared here}}
#endif
void *operator new[](size_t count, std::align_val_t al);
void operator delete[](void*, std::align_val_t);

void operator delete(void *, size_t); // expected-note 1+ {{candidate}}
#ifndef __cpp_sized_deallocation
// expected-note@-2 1+ {{non-usual 'operator delete' declared here}}
#endif
void operator delete[](void*, size_t);

// Declare some other placemenet operators.
void *operator new(size_t, void*, bool) throw(); // expected-note 1+ {{candidate function}}
void *operator new[](size_t, void*, bool) throw();

void *NP = 0;

void test_typo_in_args() {
  __builtin_operator_new(DNE);          // expected-error {{undeclared identifier 'DNE'}}
  __builtin_operator_new(DNE, DNE2);    // expected-error {{undeclared identifier 'DNE'}} expected-error {{'DNE2'}}
  __builtin_operator_delete(DNE);       // expected-error {{'DNE'}}
  __builtin_operator_delete(DNE, DNE2); // expected-error {{'DNE'}} expected-error {{'DNE2'}}
}

void test_arg_types() {
  __builtin_operator_new(NP);                      // expected-error {{no matching function for call to 'operator new'}}
  __builtin_operator_new(NP, std::align_val_t(0)); // expected-error {{no matching function for call to 'operator new'}}
}
void test_return_type() {
  int w = __builtin_operator_new(42);        // expected-error {{cannot initialize a variable of type 'int' with an rvalue of type 'void *'}}
  int y = __builtin_operator_delete(NP);     // expected-error {{cannot initialize a variable of type 'int' with an rvalue of type 'void'}}
}

void test_aligned_new() {
#ifdef __cpp_aligned_new
  void *p = __builtin_operator_new(42, std::align_val_t(2));
  __builtin_operator_delete(p, std::align_val_t(2));
#else
  // FIXME: We've manually declared the aligned new/delete overloads,
  // but LangOpts::AlignedAllocation is false. Should our overloads be considered
  // usual allocation/deallocation functions?
  void *p = __builtin_operator_new(42, std::align_val_t(2)); // expected-error {{call to '__builtin_operator_new' selects non-usual allocation function}}
  __builtin_operator_delete(p, std::align_val_t(2));         // expected-error {{call to '__builtin_operator_delete' selects non-usual deallocation function}}
#endif
}

void test_sized_delete() {
#ifdef __cpp_sized_deallocation
  __builtin_operator_delete(NP, 4);
#else
  __builtin_operator_delete(NP, 4); // expected-error {{call to '__builtin_operator_delete' selects non-usual deallocation function}}
#endif
}

void *operator new(size_t, bool);   // expected-note 1+ {{candidate}}
// expected-note@-1 {{non-usual 'operator new' declared here}}
void operator delete(void *, bool); // expected-note 1+ {{candidate}}
// expected-note@-1 {{non-usual 'operator delete' declared here}}

void test_non_usual() {
  __builtin_operator_new(42, true);     // expected-error {{call to '__builtin_operator_new' selects non-usual allocation function}}
  __builtin_operator_delete(NP, false); // expected-error {{call to '__builtin_operator_delete' selects non-usual deallocation function}}
}

template <int ID>
struct Tag {};
struct ConvertsToTypes {
  operator std::align_val_t() const;
  operator Tag<0>() const;
};

void *operator new(size_t, Tag<0>);   // expected-note 0+ {{candidate}}
void operator delete(void *, Tag<0>); // expected-note 0+ {{candidate}}

void test_ambiguous() {
#ifdef __cpp_aligned_new
  ConvertsToTypes cvt;
  __builtin_operator_new(42, cvt);    // expected-error {{call to 'operator new' is ambiguous}}
  __builtin_operator_delete(NP, cvt); // expected-error {{call to 'operator delete' is ambiguous}}
#endif
}

void test_no_args() {
  __builtin_operator_new();    // expected-error {{no matching function for call to 'operator new'}}
  __builtin_operator_delete(); // expected-error {{no matching function for call to 'operator delete'}}
}

void test_no_matching_fn() {
  Tag<1> tag;
  __builtin_operator_new(42, tag);    // expected-error {{no matching function for call to 'operator new'}}
  __builtin_operator_delete(NP, tag); // expected-error {{no matching function for call to 'operator delete'}}
}

template <class Tp, class Up, class RetT>
void test_dependent_call(Tp new_arg, Up delete_arg, RetT) {
  RetT ret = __builtin_operator_new(new_arg);
  __builtin_operator_delete(delete_arg);
}
template void test_dependent_call(int, int*, void*);

void test_const_attribute() {
  __builtin_operator_new(42); // expected-warning {{ignoring return value of function declared with const attribute}}
#ifdef __cpp_aligned_new
  __builtin_operator_new(42, std::align_val_t(8)); // expected-warning {{ignoring return value of function declared with const attribute}}
#endif
}
