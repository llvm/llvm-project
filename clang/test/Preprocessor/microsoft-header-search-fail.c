// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -Eonly -fms-compatibility %t/test.c -I %t/include -verify

//--- test.c
#include "x/header.h"
#include "z/header.h"

// expected-warning-re@include/y/header.h:1 {{#include resolved using non-portable Microsoft search rules as: {{.*}}x/culprit.h}}
// expected-error@include/z/header.h:1 {{'culprit.h' file not found}}

//--- include/x/header.h
#include "y/header.h"

//--- include/y/header.h
#include "culprit.h"

//--- include/x/culprit.h

//--- include/z/header.h
#include "culprit.h"
