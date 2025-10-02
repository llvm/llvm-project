// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,security.ArrayBound,debug.ExprInspection \
// RUN:                    -fstrict-flex-arrays=0 -DSTRICT_FLEX=0 \
// RUN:                    -verify %s
//
// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,security.ArrayBound,debug.ExprInspection \
// RUN:                    -fstrict-flex-arrays=1 -DSTRICT_FLEX=1 \
// RUN:                    -verify %s
//
// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,security.ArrayBound,debug.ExprInspection \
// RUN:                    -fstrict-flex-arrays=2 -DSTRICT_FLEX=2 \
// RUN:                    -verify %s
//
// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,security.ArrayBound,debug.ExprInspection \
// RUN:                    -fstrict-flex-arrays=2 -DSTRICT_FLEX=2 \
// RUN:                    -DWARN_FLEXIBLE_ARRAY -analyzer-config security.ArrayBound:EnableFakeFlexibleArrayWarn=true \
// RUN:                    -verify %s
//
// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,security.ArrayBound,debug.ExprInspection \
// RUN:                    -fstrict-flex-arrays=3 -DSTRICT_FLEX=3 \
// RUN:                    -verify %s

#include "../Inputs/system-header-simulator-for-malloc.h"

void clang_analyzer_warnIfReached();

struct RealFAM {
  int size;
  int args[];
};

int use_fam(struct RealFAM *ptr) {
  int x = ptr->args[1];
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return x;
}

struct FAM0 {
  int size;
  int args[0];
};

int use_fam0(struct FAM0 *ptr) {
  int x = ptr->args[1];
#if STRICT_FLEX > 2
  // expected-warning@-2 {{Out of bound access to memory after the end of the field 'args'}}
#else
#ifdef WARN_FLEXIBLE_ARRAY
  // expected-warning@-5 {{Potential out of bound access to the field 'args', which may be a 'flexible array member'}}
#endif
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
#endif
  return x;
}

struct FAM1 {
  int size;
  int args[1];
};

int use_fam1(struct FAM1 *ptr) {
  int x = ptr->args[2];
#if STRICT_FLEX > 1
  // expected-warning@-2 {{Out of bound access to memory after the end of the field 'args'}}
#else
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
#endif
  return x;
}

struct FAMN {
  int size;
  int args[64];
};

int use_famn(struct FAMN *ptr) {
  int x = ptr->args[128];
#if STRICT_FLEX > 0
  // expected-warning@-2 {{Out of bound access to memory after the end of the field 'args'}}
#else
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
#endif
  return x;
}

struct NotFAM {
  int keys[1];
  int values[1];
};

int use_not_fam(struct NotFAM *ptr) {
  return ptr->keys[3]; // expected-warning {{Out of bound access to memory after the end of the field 'keys'}}
}

// Pattern used in GCC
union FlexibleArrayUnion {
  int args[1];
  struct {
    int x, y, z;
  };
};

int use_union(union FlexibleArrayUnion *p) {
  int x = p->args[2];
#if STRICT_FLEX > 1
  // expected-warning@-2 {{Out of bound access to memory after the end of the field 'args'}}
#else
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
#endif
  return x;
}
