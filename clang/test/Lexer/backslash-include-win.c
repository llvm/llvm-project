// REQUIRES: system-windows
// RUN: mkdir -p %t/backslash
// RUN: cp %S/Inputs/case-insensitive-include.h %t/backslash/case-insensitive-include.h
// RUN: %clang_cc1 -fsyntax-only -Wnonportable-include-path-separator -I%t %s 2>&1 | FileCheck --check-prefixes=CHECK-ENABLED,CHECK-ALL %s
// RUN: %clang_cc1 -fsyntax-only -Wnonportable-include-path -Wno-nonportable-include-path-separator -I%t %s 2>&1 | FileCheck --check-prefix=CHECK-DISABLED,CHECK-ALL %s

#include "backslash\case-insensitive-include.h"
// CHECK-ENABLED: non-portable path to file
// CHECK-ENABLED: specified path contains backslashes
// CHECK-ENABLED: "backslash/case-insensitive-include.h"
// CHECK-DISABLED-NOT: non-portable path to file

// Despite fixing the same span, nonportable-include-path is still a separate diagnostic
// that can fire at the same time.
#include "backslash\CASE-insensitive-include.h"
// CHECK-ALL: non-portable path to file
// CHECK-ALL: specified path differs in case from file name on disk
// CHECK-ALL: "backslash\case-insensitive-include.h"
// CHECK-ENABLED: non-portable path to file
// CHECK-ENABLED: specified path contains backslashes
// CHECK-ENABLED: "backslash/CASE-insensitive-include.h"
// CHECK-DISABLED-NOT: non-portable path to file
