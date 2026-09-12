// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify=warn %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=warn %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=err %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -DCODEGEN -emit-llvm -o - %s | FileCheck %s

class Foo; // warn-note {{forward declaration of 'Foo'}} \
           // err-note {{forward declaration of 'Foo'}}

typedef __SIZE_TYPE__ size_t;

namespace std {
  enum class align_val_t : size_t {};
  template <class T> struct type_identity {
    typedef T type;
  };
}

void operator delete(std::type_identity<Foo>, void *, size_t, std::align_val_t); // warn-warning {{type aware allocators are a Clang extension}} \
                                                                                 // err-warning {{type aware allocators are a Clang extension}}

#ifndef CODEGEN
void f(Foo *o) {
  delete o;
  // warn-warning@-1 {{deleting pointer to incomplete type 'Foo' is incompatible with C++2c and may cause undefined behavior}}
  // warn-error@-2 {{type-aware deallocation function matches incomplete type 'Foo'; the type must be complete to use type-aware deallocation}}
  // err-error@-3 {{cannot delete pointer to incomplete type 'Foo'}}
  // err-error@-4 {{type-aware deallocation function matches incomplete type 'Foo'; the type must be complete to use type-aware deallocation}}
}
#endif

class Bar; // warn-note {{forward declaration of 'Bar'}} \
           // err-note {{forward declaration of 'Bar'}}

void g(Bar *b) {
  delete b;
  // warn-warning@-1 {{type-aware deallocation is not used for deletion of pointer to incomplete type 'Bar'}}
  // warn-warning@-2 {{deleting pointer to incomplete type 'Bar' is incompatible with C++2c and may cause undefined behavior}}
  // err-warning@-3 {{type-aware deallocation is not used for deletion of pointer to incomplete type 'Bar'}}
  // err-error@-4 {{cannot delete pointer to incomplete type 'Bar'}}
}

// CHECK-LABEL: define {{.*}} @_Z1gP3Bar
// CHECK-NOT: call {{.*}} @{{.*}}operator delete{{.*}}type_identity
// CHECK: call void @_ZdlPv
