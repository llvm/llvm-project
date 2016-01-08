// RUN: clang-tidy %s -checks='-*,google-build-namespaces,google-build-using-namespace' -header-filter='.*' -- | FileCheck %s -implicit-check-not="{{warning|error}}:"
#include "Inputs/google-namespaces.h"
// CHECK: warning: do not use unnamed namespaces in header files [google-build-namespaces]

using namespace spaaaace;
// CHECK: :[[@LINE-1]]:1: warning: do not use namespace using-directives; use using-declarations instead [google-build-using-namespace]

using spaaaace::core; // no-warning
