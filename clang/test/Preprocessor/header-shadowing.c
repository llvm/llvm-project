// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -Wheader-shadowing -Eonly %t/main.c -I %t/include  2>&1 | FileCheck %s --check-prefix=SHADOWING
// SHADOWING: {{.*}} warning: multiple candidates for header 'header.h' found;

//--- main.c
#include "header.h"

//--- include/header.h
#pragma once

//--- header.h
#pragma once
