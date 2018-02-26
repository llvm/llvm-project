// RUN: %clang_cc1 -fsyntax-only -I %S/Inputs -x objective-c -Wnullability-completeness -Werror -verify %s
// RUN: %clang_cc1 -fsyntax-only -I %S/Inputs -x objective-c -Wnullability-completeness -Werror -verify -DUSE_MUTABLE %s
// expected-no-diagnostics

#include "nullability-completeness-cferror.h"
