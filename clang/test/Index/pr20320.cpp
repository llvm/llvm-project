// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 local -x c++ %s
#include "pr20320.h"
