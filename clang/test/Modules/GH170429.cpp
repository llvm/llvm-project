// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -emit-module -o %t/A.pcm -fmodule-name=A -x c++ %t/module.modulemap -Wdelete-non-virtual-dtor
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodule-file=A=%t/A.pcm -I%t %t/use.cc -Wdelete-non-virtual-dtor -verify

// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodule-file=A=%t/A.pcm -I%t %t/use.cc -Wsystem-headers -Wdelete-non-virtual-dtor -verify=sys

//--- module.modulemap
module A [system] {
  header "A.h"
}

//--- A.h
template<typename T>
void make_unique() {
  T();
}

template<typename T>
void delete_ptr(T *p) {
  delete p;
}

int x;;

//--- use.cc
#include "A.h"

// 1. Check that deprecated warnings are emitted even if the template is in a
// system module, when the instantiation is triggered from user code.
// This works because SemaAvailability uses AllowWarningInSystemHeaders RAII
// to temporarily disable suppression.
// expected-warning@A.h:3 2 {{'C' is deprecated}}
// sys-warning@A.h:3 2 {{'C' is deprecated}}

// 2. Check that warnings with ShowInSystemHeader (like -Wdelete-non-virtual-dtor)
// are still emitted from system modules.
// expected-warning@A.h:8 {{delete called on non-final 'Base' that has virtual functions but non-virtual destructor}}
// sys-warning@A.h:8 {{delete called on non-final 'Base' that has virtual functions but non-virtual destructor}}

// 3. Check that unrelated system header warnings (like -Wextra-semi) remain
// suppressed even with -Wsystem-headers for explicit modules, preserving
// the hermeticity of the diagnostic state serialized into the PCM.
// If this were not hermetic, we would see a 'sys-warning' for the extra semi
// at A.h:11.

class C {
public:
  C() __attribute__((deprecated("",""))); // expected-note 2 {{'C' has been explicitly marked deprecated here}} \
                                          // sys-note 2 {{'C' has been explicitly marked deprecated here}}
};

struct Base {
  virtual void f();
};
struct Derived : Base {};

void bar() {
  make_unique<C>(); // expected-note {{in instantiation of function template specialization 'make_unique<C>' requested here}} \
                    // sys-note {{in instantiation of function template specialization 'make_unique<C>' requested here}}
  delete_ptr((Base*)new Derived); // expected-note {{in instantiation of function template specialization 'delete_ptr<Base>' requested here}} \
                                  // sys-note {{in instantiation of function template specialization 'delete_ptr<Base>' requested here}}
}
