// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-cross-tu-misplaced-lifetimebound -Wno-dangling -I%t -verify=cross %t/cross.cpp

//--- cross.h
struct HeaderObj {
  ~HeaderObj() {}
};

HeaderObj &header_param(HeaderObj &obj); // cross-warning {{'lifetimebound' attribute on a cross-TU definition is not visible to callers; add it to the declaration instead}}

struct HeaderS {
  HeaderObj data;
  HeaderObj &header_this(); // cross-warning {{'lifetimebound' attribute on a cross-TU definition is not visible to callers; add it to the declaration instead}}
};

//--- cross.cpp
#include "cross.h"

HeaderObj &header_param(HeaderObj &obj [[clang::lifetimebound]]) { // cross-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

HeaderObj &HeaderS::header_this() [[clang::lifetimebound]] { // cross-note {{'lifetimebound' attribute appears here on the definition}}
  return data;
}
