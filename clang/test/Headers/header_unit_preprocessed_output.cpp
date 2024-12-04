// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -verify -std=c++20 -emit-header-unit -xc++-user-header bz0.h
// RUN: %clang_cc1 -verify -std=c++20 -fmodule-file=bz0.pcm -xc++-user-header bz1.h -E -o bz1.output.h
// RUN: FileCheck %s < bz1.output.h
// RUN: %clang_cc1 -std=c++20 -fmodule-file=bz0.pcm -emit-header-unit -xc++-user-header bz1.output.h

//--- bz0.h
// expected-no-diagnostics
#pragma once

void foo();

//--- bz1.h
// expected-no-diagnostics
import "bz0.h";

// CHECK: # 1 ".{{/|\\\\?}}bz1.h"
// CHECK: import ".{{/|\\\\?}}bz0.h";
