#include "crash-preamble-classes.h"

struct Z : Y {
  Z() {}
};

// RUN: env CINDEXTEST_EDITING=1 \
// RUN: c-index-test -test-load-source-reparse 5 local -I %S/Inputs %s
