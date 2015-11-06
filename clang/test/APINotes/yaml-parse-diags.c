// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fapinotes -fapinotes-cache-path=%t %s -I %S/Inputs/BrokenHeaders -verify

#include "SomeBrokenLib.h"

// expected-error@APINotes.apinotes:4{{unknown key 'Nu llabilityOfRet'}}
