// RUN: %clang_cc1 %s -verify -I %S/Inputs

#include "//leading-double-slash.h" // expected-error {{'//leading-double-slash.h' file not found}}
