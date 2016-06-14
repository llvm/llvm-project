// REQUIRES: case-insensitive-filesystem

// RUN: mkdir -p %T/apath
// RUN: mkdir -p %T/asystempath
// RUN: cp %S/Inputs/case-insensitive-include.h %T
// RUN: cp %S/Inputs/case-insensitive-include.h %T/asystempath/case-insensitive-include2.h
// RUN: cd %T
// RUN: %clang_cc1 -fsyntax-only %s -include %s -I %T -isystem %T/asystempath -verify
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s -include %s -I %T -isystem %T/asystempath 2>&1 | FileCheck %s

// Known standard header, so warn:
#include <StdDef.h> // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:20}:"<stddef.h>"

#include "case-insensitive-include.h"
#include "Case-Insensitive-Include.h" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:38}:"\"case-insensitive-include.h\""

#include "../Output/./case-insensitive-include.h"
#include "../Output/./Case-Insensitive-Include.h" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:50}:"\"../Output/./case-insensitive-include.h\""
#include "../output/./case-insensitive-include.h" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:50}:"\"../Output/./case-insensitive-include.h\""

#include "apath/.././case-insensitive-include.h"
#include "apath/.././Case-Insensitive-Include.h" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:49}:"\"apath/.././case-insensitive-include.h\""
#include "APath/.././case-insensitive-include.h" // For the sake of efficiency, this case is not diagnosed. :-(

#include "../Output/./apath/.././case-insensitive-include.h"
#include "../Output/./APath/.././case-insensitive-include.h" // For the sake of efficiency, this case is not diagnosed. :-(
#include "../output/./apath/.././case-insensitive-include.h" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:61}:"\"../Output/./apath/.././case-insensitive-include.h\""

#include "CASE-INSENSITIVE-INCLUDE2.H" // Found in an -isystem directory. No warning.
