// RUN: %clang_cc1 -E -header-include-format=json -header-include-filtering=only-direct-system -header-include-file %t.txt -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s
// RUN: cat %t.txt | FileCheck %s --check-prefix=SUPPORTED

// RUN: not %clang_cc1 -E -header-include-format=textual -header-include-filtering=only-direct-system -header-include-file %t.txt -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED0
// RUN: not %clang_cc1 -E -header-include-format=json -header-include-filtering=none -header-include-file %t.txt -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED1
// RUN: env CC_PRINT_HEADERS_FORMAT=textual CC_PRINT_HEADERS_FILTERING=only-direct-system CC_PRINT_HEADERS_FILE=%t.txt not %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED2
// RUN: env CC_PRINT_HEADERS_FORMAT=json CC_PRINT_HEADERS_FILTERING=none CC_PRINT_HEADERS_FILE=%t.txt not %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED3
// RUN: env CC_PRINT_HEADERS_FORMAT=json CC_PRINT_HEADERS_FILE=%t.txt not %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED4
// RUN: not %clang_cc1 -E -header-include-filtering=only-direct-system -header-include-file %t.txt -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED5
// RUN: not %clang_cc1 -E -header-include-format=json -header-include-file %t.txt -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED6

// RUN: rm %t.txt
// RUN: env CC_PRINT_HEADERS_FORMAT=json CC_PRINT_HEADERS_FILTERING=only-direct-system CC_PRINT_HEADERS_FILE=%t.txt %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null
// RUN: cat %t.txt | FileCheck %s --check-prefix=SUPPORTED

// RUN: rm %t.txt
// RUN: env CC_PRINT_HEADERS_FORMAT=json CC_PRINT_HEADERS_FILTERING=direct-per-file CC_PRINT_HEADERS_FILE=%t.txt %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system %s -o /dev/null
// RUN: cat %t.txt | FileCheck %s --check-prefix=SUPPORTED_PERFILE

#include "system0.h"
#include "header0.h"
#include "system2.h"

// SUPPORTED: {"source":"{{[^,]*}}print-header-json.c","includes":["{{[^,]*}}system0.h","{{[^,]*}}system3.h","{{[^,]*}}system2.h"]}
// SUPPORTED_PERFILE: [{"source":"{{[^,]*}}print-header-json.c","includes":["{{[^,]*}}system0.h","{{[^,]*}}header0.h","{{[^,]*}}system2.h"]},{"source":"{{[^,]*}}header0.h","includes":["{{[^,]*}}system3.h","{{[^,]*}}header1.h","{{[^,]*}}header2.h"]}]

// UNSUPPORTED0: error: unsupported combination: -header-include-format=textual and -header-include-filtering=only-direct-system
// UNSUPPORTED1: error: unsupported combination: -header-include-format=json and -header-include-filtering=none
// UNSUPPORTED2: error: unsupported combination: CC_PRINT_HEADERS_FORMAT=textual and CC_PRINT_HEADERS_FILTERING=only-direct-system
// UNSUPPORTED3: error: unsupported combination: CC_PRINT_HEADERS_FORMAT=json and CC_PRINT_HEADERS_FILTERING=none
// UNSUPPORTED4: error: environment variable CC_PRINT_HEADERS_FORMAT=json requires a compatible value for CC_PRINT_HEADERS_FILTERING
// UNSUPPORTED5: error: -header-include-filtering=only-direct-system requires a compatible value for -header-include-format
// UNSUPPORTED6: error: -header-include-format=json requires a compatible value for -header-include-filtering
