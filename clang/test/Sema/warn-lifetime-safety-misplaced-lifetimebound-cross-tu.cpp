// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-cross-tu-misplaced-lifetimebound -Wno-dangling -I%t -verify %t/cross.cpp

//--- cross.h
struct HeaderObj {
  ~HeaderObj() {}
};

HeaderObj &header_param(HeaderObj &obj); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers in other translation units; add it to the declaration instead}}

struct HeaderS {
  HeaderObj data;
  HeaderObj &header_this(); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers in other translation units; add it to the declaration instead}}
};

//--- cross.cpp
#include "cross.h"

HeaderObj &header_param(HeaderObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

HeaderObj &HeaderS::header_this() [[clang::lifetimebound]] { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return data;
}
