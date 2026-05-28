// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-cross-tu-misplaced-lifetimebound -Wno-dangling -I%t -verify %t/cross.cpp
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-cross-tu-misplaced-lifetimebound -Wno-dangling -I%t -fdiagnostics-parseable-fixits %t/cross.cpp 2>&1 | FileCheck %s
// RUN: %clang_cc1 -Wlifetime-safety-cross-tu-misplaced-lifetimebound -Wno-dangling -I%t -fixit %t/cross.cpp
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-cross-tu-misplaced-lifetimebound -Wno-dangling -I%t -Werror %t/cross.cpp

//--- cross.h
struct HeaderObj {
  ~HeaderObj() {}
};

HeaderObj &header_param(HeaderObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers in other translation units; add it to the declaration instead}}
                                     // CHECK: cross.h:[[#PARAM_WARN_LINE:]]:{{[0-9]+}}: warning: 'lifetimebound' attribute on this definition is not visible
                        obj          // CHECK: fix-it:"{{.*}}cross.h":{[[#PARAM_WARN_LINE+2]]:{{[0-9]+}}-[[#PARAM_WARN_LINE+2]]:{{[0-9]+}}}:" {{\[\[}}clang::lifetimebound{{\]\]}}"
                        );

struct HeaderS {
  HeaderObj data;
  HeaderObj &header_this(
                         );  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers in other translation units; add it to the declaration instead}}
                             // CHECK: cross.h:[[#THIS_WARN_LINE:]]:{{[0-9]+}}: warning: 'lifetimebound' attribute on this definition is not visible
                             // CHECK: fix-it:"{{.*}}cross.h":{[[#THIS_WARN_LINE]]:{{[0-9]+}}-[[#THIS_WARN_LINE]]:{{[0-9]+}}}:" {{\[\[}}clang::lifetimebound{{\]\]}}"
};

//--- cross_1.h
struct HeaderObj;
HeaderObj &multi_header_param(HeaderObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers in other translation units; add it to the declaration instead}}
                                           // CHECK: cross_1.h:[[#PARAM_WARN_LINE:]]:{{[0-9]+}}: warning: 'lifetimebound' attribute on this definition is not visible
                              obj          // CHECK: fix-it:"{{.*}}cross_1.h":{[[#PARAM_WARN_LINE+2]]:{{[0-9]+}}-[[#PARAM_WARN_LINE+2]]:{{[0-9]+}}}:" {{\[\[}}clang::lifetimebound{{\]\]}}"
                              );

//--- cross_2.h
struct HeaderObj;
HeaderObj &multi_header_param(HeaderObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers in other translation units; add it to the declaration instead}}
                                           // CHECK: cross_2.h:[[#PARAM_WARN_LINE:]]:{{[0-9]+}}: warning: 'lifetimebound' attribute on this definition is not visible
                              obj          // CHECK: fix-it:"{{.*}}cross_2.h":{[[#PARAM_WARN_LINE+2]]:{{[0-9]+}}-[[#PARAM_WARN_LINE+2]]:{{[0-9]+}}}:" {{\[\[}}clang::lifetimebound{{\]\]}}"
                              );

//--- cross.cpp
#include "cross.h"
#include "cross_1.h"
#include "cross_2.h"

HeaderObj &header_param(HeaderObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

HeaderObj &HeaderS::header_this() [[clang::lifetimebound]] { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return data;
}

HeaderObj &multi_header_param(HeaderObj &obj [[clang::lifetimebound]]) { // expected-note 2 {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}
