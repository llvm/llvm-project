// REQUIRES: case-insensitive-filesystem

// Test this without pch.
// RUN: mkdir -p %t-dir
// RUN: cp %S/Inputs/case-insensitive-include.h %t-dir
// RUN: %clang_cc1 -Wno-nonportable-include-path -fsyntax-only %s -include %s -I %t-dir -verify

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t.pch %s -I %t-dir

// Modify inode of the header.
// RUN: cp %t-dir/case-insensitive-include.h %t.copy
// RUN: touch -r %t-dir/case-insensitive-include.h %t.copy
// RUN: mv %t.copy %t-dir/case-insensitive-include.h

// RUN: %clang_cc1 -Wno-nonportable-include-path -fsyntax-only %s -include-pch %t.pch -I %t-dir -verify

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#include "case-insensitive-include.h"
#include "Case-Insensitive-Include.h"

#else

#include "Case-Insensitive-Include.h"

#endif
