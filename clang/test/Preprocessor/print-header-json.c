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

// RUN: rm %t.txt
// RUN: rm -rf %t
// RUN: env CC_PRINT_HEADERS_FORMAT=json CC_PRINT_HEADERS_FILTERING=direct-per-file CC_PRINT_HEADERS_FILE=%t.txt %clang -fsyntax-only -I %S/Inputs/print-header-json -isystem %S/Inputs/print-header-json/system -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -o /dev/null
// RUN: cat %t.txt | FileCheck %s --check-prefix=SUPPORTED_PERFILE_MODULES

// SUPPORTED: {"source":"{{[^,]*}}print-header-json.c","includes":["{{[^,]*}}system0.h","{{[^,]*}}system3.h","{{[^,]*}}system2.h"]}
// SUPPORTED_PERFILE: {"version":"2.0.0","dependencies":[{"source":"{{[^,]*}}print-header-json.c","includes":[{"location":"{{[^,]*}}print-header-json.c:20:1","file":"{{[^,]*}}system0.h"},{"location":"{{[^,]*}}print-header-json.c:21:1","file":"{{[^,]*}}header0.h"},{"location":"{{[^,]*}}print-header-json.c:22:1","file":"{{[^,]*}}system2.h"}],"imports":[]},{"source":"{{[^,]*}}header0.h","includes":[{"location":"{{[^,]*}}header0.h:1:1","file":"{{[^,]*}}system3.h"},{"location":"{{[^,]*}}header0.h:2:1","file":"{{[^,]*}}header1.h"},{"location":"{{[^,]*}}header0.h:3:1","file":"{{[^,]*}}header2.h"}],"imports":[]}]}
// SUPPORTED_PERFILE_MODULES: {"version":"2.0.0","dependencies":[{"source":"{{[^,]*}}print-header-json.c","includes":[{"location":"{{[^,]*}}print-header-json.c:20:1","file":"{{[^,]*}}system0.h"}],"imports":[{"location":"{{[^,]*}}print-header-json.c:21:1","module":"module0","file":"{{[^,]*}}print-header-json{{[\/\\]+}}module.modulemap"},{"location":"{{[^,]*}}print-header-json.c:22:1","module":"systemmodule0","file":"{{[^,]*}}print-header-json{{[\/\\]+}}system{{[\/\\]+}}module.modulemap"}]}]}

// UNSUPPORTED0: error: unsupported combination: -header-include-format=textual and -header-include-filtering=only-direct-system
// UNSUPPORTED1: error: unsupported combination: -header-include-format=json and -header-include-filtering=none
// UNSUPPORTED2: error: unsupported combination: CC_PRINT_HEADERS_FORMAT=textual and CC_PRINT_HEADERS_FILTERING=only-direct-system
// UNSUPPORTED3: error: unsupported combination: CC_PRINT_HEADERS_FORMAT=json and CC_PRINT_HEADERS_FILTERING=none
// UNSUPPORTED4: error: environment variable CC_PRINT_HEADERS_FORMAT=json requires a compatible value for CC_PRINT_HEADERS_FILTERING
// UNSUPPORTED5: error: -header-include-filtering=only-direct-system requires a compatible value for -header-include-format
// UNSUPPORTED6: error: -header-include-format=json requires a compatible value for -header-include-filtering
