// RUN: %clang_cc1 %s -verify -I %S/Inputs
// REQUIRES: system-windows

#include "\\leading-double-slash.h" // expected-error {{'\\leading-double-slash.h' file not found}}
