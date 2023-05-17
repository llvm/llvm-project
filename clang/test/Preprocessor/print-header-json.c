// RUN: %clang_cc1 -E -header-include-format=json -header-include-filtering=only-direct-system -header-include-file %t.txt -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s
// RUN: cat %t.txt | FileCheck %s --check-prefix=SUPPORTED
// RUN: not %clang_cc1 -E -header-include-format=textual -header-include-filtering=only-direct-system -header-include-file %t.txt -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED0
// RUN: not %clang_cc1 -E -header-include-format=json -header-include-filtering=none -header-include-file %t.txt -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED1
// RUN: rm %t.txt
// RUN: env CC_PRINT_HEADERS_FORMAT=json CC_PRINT_HEADERS_FILTERING=only-direct-system CC_PRINT_HEADERS_FILE=%t.txt %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null
// RUN: env CC_PRINT_HEADERS_FORMAT=textual CC_PRINT_HEADERS_FILTERING=only-direct-system CC_PRINT_HEADERS_FILE=%t.txt not %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED2
// RUN: env CC_PRINT_HEADERS_FORMAT=json CC_PRINT_HEADERS_FILTERING=none CC_PRINT_HEADERS_FILE=%t.txt not %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED3
// RUN: cat %t.txt | FileCheck %s --check-prefix=SUPPORTED

#include "system0.h"
#include "header0.h"
#include "system2.h"

// SUPPORTED: {"source":"{{[^,]*}}print-header-json.c","includes":["{{[^,]*}}system0.h","{{[^,]*}}system3.h","{{[^,]*}}system2.h"]}

// UNSUPPORTED0: error: unsupported combination: -header-include-format=textual and -header-include-filtering=only-direct-system
// UNSUPPORTED1: error: unsupported combination: -header-include-format=json and -header-include-filtering=none
// UNSUPPORTED2: error: unsupported combination: CC_PRINT_HEADERS_FORMAT=textual and CC_PRINT_HEADERS_FILTERING=only-direct-system
// UNSUPPORTED3: error: unsupported combination: CC_PRINT_HEADERS_FORMAT=json and CC_PRINT_HEADERS_FILTERING=none
