// REQUIRES: case-insensitive-filesystem

// RUN: mkdir -p %t/Output/apath
// RUN: cp %S/Inputs/case-insensitive-include.h %t/Output
// RUN: cd %t/Output
// RUN: %clang_cc1 -fsyntax-only -fms-compatibility %s -include %s -I %t/Output -verify
// RUN: %clang_cc1 -fsyntax-only -fms-compatibility -fdiagnostics-parseable-fixits %s -include %s -I %t/Output 2>&1 | FileCheck %s

#include "..\Output\.\case-insensitive-include.h"
#include "..\Output\.\Case-Insensitive-Include.h" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:50}:"\"..\\Output\\.\\case-insensitive-include.h\""
#include "..\output\.\case-insensitive-include.h" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:50}:"\"..\\Output\\.\\case-insensitive-include.h\""

#include "apath\..\.\case-insensitive-include.h"
#include "apath\..\.\Case-Insensitive-Include.h" // expected-warning {{non-portable path}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:49}:"\"apath\\..\\.\\case-insensitive-include.h\""
#include "APath\..\.\case-insensitive-include.h" // For the sake of efficiency, this case is not diagnosed. :-(
