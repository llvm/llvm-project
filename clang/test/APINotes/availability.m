// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -Wno-private-module -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify

#include "HeaderLib.h"
#import <SomeKit/SomeKit.h>
#import <SomeKit/SomeKit_Private.h>

int main() {
  int i;
  i = unavailable_function(); // expected-error{{'unavailable_function' is unavailable: I beg you not to use this}}
  // expected-note@HeaderLib.h:8{{'unavailable_function' has been explicitly marked unavailable here}}
  i = unavailable_global_int; // expected-error{{'unavailable_global_int' is unavailable}}
  // expected-note@HeaderLib.h:9{{'unavailable_global_int' has been explicitly marked unavailable here}}

  unavailable_typedef t; // expected-error{{'unavailable_typedef' is unavailable}}
  // expected-note@HeaderLib.h:14{{'unavailable_typedef' has been explicitly marked unavailable here}}

  struct unavailable_struct s; // expected-error{{'unavailable_struct' is unavailable}}
  // expected-note@HeaderLib.h:15{{'unavailable_struct' has been explicitly marked unavailable here}}

  B *b = 0; // expected-error{{'B' is unavailable: just don't}}
  // expected-note@SomeKit/SomeKit.h:15{{'B' has been explicitly marked unavailable here}}

  id<InternalProtocol> proto = 0; // expected-error{{'InternalProtocol' is unavailable: not for you}}
  // expected-note@SomeKit/SomeKit_Private.h:12{{'InternalProtocol' has been explicitly marked unavailable here}}

  A *a = 0;
  i = a.intValue; // expected-error{{intValue' is unavailable: wouldn't work anyway}}
  // expected-note@SomeKit/SomeKit.h:12{{'intValue' has been explicitly marked unavailable here}}

  [a transform:a]; // expected-error{{'transform:' is unavailable: anything but this}}
  // expected-note@SomeKit/SomeKit.h:6{{'transform:' has been explicitly marked unavailable here}}

  [a implicitGetOnlyInstance]; // expected-error{{'implicitGetOnlyInstance' is unavailable: getter gone}}
  // expected-note@SomeKit/SomeKit.h:53{{'implicitGetOnlyInstance' has been explicitly marked unavailable here}}
  [A implicitGetOnlyClass]; // expected-error{{'implicitGetOnlyClass' is unavailable: getter gone}}
  // expected-note@SomeKit/SomeKit.h:54{{'implicitGetOnlyClass' has been explicitly marked unavailable here}}
  [a implicitGetSetInstance]; // expected-error{{'implicitGetSetInstance' is unavailable: getter gone}}
  // expected-note@SomeKit/SomeKit.h:56{{'implicitGetSetInstance' has been explicitly marked unavailable here}}
  [a setImplicitGetSetInstance: a];  // expected-error{{'setImplicitGetSetInstance:' is unavailable: setter gone}}
  // expected-note@SomeKit/SomeKit.h:56{{'setImplicitGetSetInstance:' has been explicitly marked unavailable here}}
  [A implicitGetSetClass]; // expected-error{{'implicitGetSetClass' is unavailable: getter gone}}
  // expected-note@SomeKit/SomeKit.h:57{{'implicitGetSetClass' has been explicitly marked unavailable here}}
  [A setImplicitGetSetClass: a];  // expected-error{{'setImplicitGetSetClass:' is unavailable: setter gone}}
  // expected-note@SomeKit/SomeKit.h:57{{'setImplicitGetSetClass:' has been explicitly marked unavailable here}}
  return 0;
}

