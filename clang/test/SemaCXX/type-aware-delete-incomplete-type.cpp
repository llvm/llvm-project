// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify=warn %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=warn %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=err %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

class Foo; // warn-note {{forward declaration of 'Foo'}} \
           // err-note {{forward declaration of 'Foo'}}

typedef __SIZE_TYPE__ size_t;

namespace std {
  enum class align_val_t : size_t {};
  template <class T> struct type_identity {
    typedef T type;
  };
}

template <class T>
void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t); // warn-warning {{type aware allocators are a Clang extension}} \
                                                                                 // err-warning {{type aware allocators are a Clang extension}}

void f(Foo *o) {
  delete o;
  // warn-warning@-1 {{type-aware deallocation is not used for deletion of pointer to incomplete type 'Foo'}}
  // warn-warning@-2 {{deleting pointer to incomplete type 'Foo' is incompatible with C++2c and may cause undefined behavior}}
  // err-warning@-3 {{type-aware deallocation is not used for deletion of pointer to incomplete type 'Foo'}}
  // err-error@-4 {{cannot delete pointer to incomplete type 'Foo'}}
}

// CHECK-LABEL: define {{.*}} @_Z1fP3Foo
// CHECK-NOT: call {{.*}} @{{.*}}operator delete{{.*}}type_identity
// CHECK: call void @_ZdlPv