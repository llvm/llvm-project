// RUN: clang-tidy %s -checks='-*,misc-anonymous-namespace-in-header,google-build-using-namespace' -header-filter='.*' -- | FileCheck %s -implicit-check-not="{{warning|error}}:"
#include "Inputs/anon-namespaces.h"
// CHECK: warning: do not use unnamed namespaces in header files [misc-anonymous-namespace-in-header]
