// RUN: %check_clang_tidy -std=c++14-or-later \
// RUN:   -check-header %S/Inputs/macro-to-enum/linkage-wrapper.h \
// RUN:   -check-header %S/Inputs/macro-to-enum/linkage-values.h \
// RUN:   -check-header %S/Inputs/macro-to-enum/linkage-local.h \
// RUN:   %s modernize-macro-to-enum %t -- \
// RUN:   -- -I%S/Inputs/macro-to-enum -fno-delayed-template-parsing
// RUN: %clang -std=c++14 -fsyntax-only -I%t.cpp.headers %t.cpp

// Nested linkage blocks and repeated header inclusion are transparent.
#include "linkage-wrapper.h"
#include "linkage-wrapper.h"

extern "C" {

#define LINKAGE_ONE 1
#define LINKAGE_TWO 2
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'LINKAGE_ONE' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'LINKAGE_TWO' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: LINKAGE_ONE = 1,
// CHECK-FIXES-NEXT: LINKAGE_TWO = 2
// CHECK-FIXES-NEXT: };

int linkage_use() {
  return LINKAGE_ONE + LINKAGE_TWO + HEADER_SEEK_CUR + HEADER_SEEK_END +
         HEADER_SEEK_SET;
}

// Declarations inside the linkage block still exclude local definitions.
int local_definition() {
#define FUNCTION_LOCAL 3
  // CHECK-FIXES: #define FUNCTION_LOCAL 3
  return FUNCTION_LOCAL;
}

int local_header_definition() {
#include "linkage-local.h"
  return LINKAGE_HEADER_LOCAL;
}

struct LinkageClass {
#define CLASS_LOCAL 4
  // CHECK-FIXES: #define CLASS_LOCAL 4
  int Value;
};

}

// The enum names remain visible after the linkage block.
int Following = LINKAGE_ONE + HEADER_SEEK_CUR;
int LocalUses = FUNCTION_LOCAL + LINKAGE_HEADER_LOCAL + CLASS_LOCAL;

// An unbraced linkage specification must still inspect its function body.
extern "C" int unbraced_definition() {
#define UNBRACED_LOCAL 5
  // CHECK-FIXES: #define UNBRACED_LOCAL 5
  return UNBRACED_LOCAL;
}

int UnbracedUse = UNBRACED_LOCAL;

namespace Hidden {
extern "C" {
#define NAMESPACE_LOCAL 6
// CHECK-FIXES: #define NAMESPACE_LOCAL 6
}
}

int NamespaceUse = NAMESPACE_LOCAL;

extern "C++" {

#define CXX_LINKAGE 7
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'CXX_LINKAGE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: CXX_LINKAGE = 7
// CHECK-FIXES-NEXT: };

template <typename T> int template_definition(T) {
#define TEMPLATE_LOCAL 8
  // CHECK-FIXES: #define TEMPLATE_LOCAL 8
  return TEMPLATE_LOCAL + CXX_LINKAGE;
}

}

int TemplateUse = template_definition(0) + TEMPLATE_LOCAL;
