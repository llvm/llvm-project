// RUN: rm -rf %t
// RUN: %clang -fmodule-maps -fmodules-cache-path=%t -fmodules-decluse -fmodule-name=XS -I %S/Inputs/declare-use -S %s -Xclang -verify
// expected-no-diagnostics

#include "s.h"
