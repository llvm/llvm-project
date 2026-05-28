// Test UTF8 BOM at start of file
// RUN: cat %S/Inputs/bom-directives.c | od -t x1 | grep -iq 'ef[[:space:]]*bb[[:space:]]*bf'
// RUN: %clang_cc1 -DTEST -print-dependency-directives-minimized-source %S/Inputs/bom-directives.c > %t 2>&1
// RUN: FileCheck %s -input-file %t
// RUN: %clang_cc1 -Eonly %t.min.c

﻿// CHECK:      #ifdef TEST
// CHECK-NEXT: #include <string>
// CHECK-NEXT: #endif
