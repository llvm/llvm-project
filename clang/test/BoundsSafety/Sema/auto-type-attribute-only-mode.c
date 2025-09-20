// RUN: %clang_cc1 -fbounds-safety -verify=bs %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=bs %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c -verify=bsa %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c++ -verify=bsa,bsa-cxx %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -verify=bsa %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c++ -verify=bsa,bsa-cxx %s

#include <ptrcheck.h>

void cb(int *__counted_by(len) ptr, int len) {
  // bs-error@+2{{passing '__counted_by' pointer as '__auto_type' initializer is not yet supported}}
  // bsa-warning@+1{{passing '__counted_by' pointer as '__auto_type' initializer is not yet supported}}
  __auto_type p = ptr;

#ifdef __cplusplus
  // bsa-cxx-warning@+1{{passing '__counted_by' pointer as 'auto' initializer is not yet supported}}
  auto q = ptr;

  // bsa-cxx-warning@+1{{passing '__counted_by' pointer as 'decltype(auto)' initializer is not yet supported}}
  decltype(auto) r = ptr;
#endif
}

void cbn(int *__counted_by_or_null(len) ptr, int len) {
  // bs-error@+2{{passing '__counted_by_or_null' pointer as '__auto_type' initializer is not yet supported}}
  // bsa-warning@+1{{passing '__counted_by_or_null' pointer as '__auto_type' initializer is not yet supported}}
  __auto_type p = ptr;

#ifdef __cplusplus
  // bsa-cxx-warning@+1{{passing '__counted_by_or_null' pointer as 'auto' initializer is not yet supported}}
  auto q = ptr;

  // bsa-cxx-warning@+1{{passing '__counted_by_or_null' pointer as 'decltype(auto)' initializer is not yet supported}}
  decltype(auto) r = ptr;
#endif
}

void sb(void *__sized_by(size) ptr, int size) {
  // bs-error@+2{{passing '__sized_by' pointer as '__auto_type' initializer is not yet supported}}
  // bsa-warning@+1{{passing '__sized_by' pointer as '__auto_type' initializer is not yet supported}}
  __auto_type p = ptr;

#ifdef __cplusplus
  // bsa-cxx-warning@+1{{passing '__sized_by' pointer as 'auto' initializer is not yet supported}}
  auto q = ptr;

  // bsa-cxx-warning@+1{{passing '__sized_by' pointer as 'decltype(auto)' initializer is not yet supported}}
  decltype(auto) r = ptr;
#endif
}

void eb(void *__ended_by(e) ptr, void *e) {
  // bs-error@+2{{passing '__ended_by' pointer as '__auto_type' initializer is not yet supported}}
  // bsa-warning@+1{{passing '__ended_by' pointer as '__auto_type' initializer is not yet supported}}
  __auto_type p = ptr;

#ifdef __cplusplus
  // bsa-cxx-warning@+1{{passing '__ended_by' pointer as 'auto' initializer is not yet supported}}
  auto q = ptr;

  // bsa-cxx-warning@+1{{passing '__ended_by' pointer as 'decltype(auto)' initializer is not yet supported}}
  decltype(auto) r = ptr;
#endif
}

void eb_end(void *__ended_by(e) ptr, void *e) {
  // bs-error@+2{{passing end pointer as '__auto_type' initializer is not yet supported}}
  // bsa-warning@+1{{passing end pointer as '__auto_type' initializer is not yet supported}}
  __auto_type p = e;

#ifdef __cplusplus
  // bsa-cxx-warning@+1{{passing end pointer as 'auto' initializer is not yet supported}}
  auto q = e;

  // bsa-cxx-warning@+1{{passing end pointer as 'decltype(auto)' initializer is not yet supported}}
  decltype(auto) r = e;
#endif
}

void cb_const(int *__counted_by(42) ptr) {
  // bs-error@+2{{passing '__counted_by' pointer as '__auto_type' initializer is not yet supported}}
  // bsa-warning@+1{{passing '__counted_by' pointer as '__auto_type' initializer is not yet supported}}
  __auto_type p = ptr;

#ifdef __cplusplus
  // bsa-cxx-warning@+1{{passing '__counted_by' pointer as 'auto' initializer is not yet supported}}
  auto q = ptr;

  // bsa-cxx-warning@+1{{passing '__counted_by' pointer as 'decltype(auto)' initializer is not yet supported}}
  decltype(auto) r = ptr;
#endif
}

#ifdef __cplusplus
template<typename T>
struct cxx_dep {
  int len;
  T __counted_by(len) ptr;

  void f() {
    // bsa-cxx-warning@+1{{passing '__counted_by' pointer as '__auto_type' initializer is not yet supported}}
    __auto_type p = ptr;

    // bsa-cxx-warning@+1{{passing '__counted_by' pointer as 'auto' initializer is not yet supported}}
    auto q = ptr;

    // bsa-cxx-warning@+1{{passing '__counted_by' pointer as 'decltype(auto)' initializer is not yet supported}}
    decltype(auto) r = ptr;
  }
};

// bsa-cxx-note@+1{{in instantiation of member function 'cxx_dep<int *>::f' requested here}}
template struct cxx_dep<int *>;
#endif
