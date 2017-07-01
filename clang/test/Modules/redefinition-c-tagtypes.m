// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -fsyntax-only %s -fmodules -fmodules-cache-path=%t.cache \
// RUN:   -fimplicit-module-maps -F%S/Inputs -verify
// RUN: %clang_cc1 -fsyntax-only %s -fmodules -fmodules-cache-path=%t.cache \
// RUN:   -fimplicit-module-maps -F%S/Inputs -DCHANGE_TAGS -verify
#include "F/F.h"

#ifndef CHANGE_TAGS
// expected-no-diagnostics
#endif

struct NS {
  int a;
#ifndef CHANGE_TAGS
  int b;
#else
  int c; // expected-note {{field has name 'c' here}}
  // expected-error@redefinition-c-tagtypes.m:12 {{type 'struct NS' has incompatible definitions}}
  // expected-note@Inputs/F.framework/PrivateHeaders/NS.h:3 {{field has name 'b' here}}
#endif
};

enum NSE {
  FST = 22,
#ifndef CHANGE_TAGS
  SND = 43,
#else
  SND = 44, // expected-note {{enumerator 'SND' with value 44 here}}
  // expected-error@redefinition-c-tagtypes.m:23 {{type 'enum NSE' has incompatible definitions}}
  // expected-note@Inputs/F.framework/PrivateHeaders/NS.h:8 {{enumerator 'SND' with value 43 here}}
#endif
  TRD = 55
};

#define NS_ENUM(_type, _name) \
  enum _name : _type _name;   \
  enum _name : _type

typedef NS_ENUM(int, NSMyEnum) {
  MinX = 11,
#ifndef CHANGE_TAGS
  MinXOther = MinX,
#else
  MinXOther = TRD, // expected-note {{enumerator 'MinXOther' with value 55 here}}
  // expected-error@redefinition-c-tagtypes.m:39 {{type 'enum NSMyEnum' has incompatible definitions}}
  // expected-note@Inputs/F.framework/PrivateHeaders/NS.h:18 {{enumerator 'MinXOther' with value 11 here}}
#endif
};
