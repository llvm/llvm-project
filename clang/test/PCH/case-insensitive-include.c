// REQUIRES: case-insensitive-filesystem

// Test this without pch.
// RUN: cp %S/Inputs/case-insensitive-include.h %T
// RUN: %clang_cc1 -Wno-nonportable-include-path -fsyntax-only %s -include %s -I %T -verify

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t.pch %s -I %T

// Modify inode of the header.
// RUN: cp %T/case-insensitive-include.h %t.copy
// RUN: touch -r %T/case-insensitive-include.h %t.copy
// RUN: mv %t.copy %T/case-insensitive-include.h

// RUN: %clang_cc1 -fsyntax-only %s -include-pch %t.pch -I %T -verify

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#include "case-insensitive-include.h"
#include "Case-Insensitive-Include.h"

#else

#include "Case-Insensitive-Include.h"

#endif
