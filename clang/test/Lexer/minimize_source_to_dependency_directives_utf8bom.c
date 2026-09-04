// Test UTF8 BOM at start of file
// RUN: cat %S/Inputs/bom-directives.c | od -t x1 | grep -iq 'ef[[:space:]]*bb[[:space:]]*bf'
// RUN: %clang_cc1 -DTEST -print-dependency-directives-minimized-source %S/Inputs/bom-directives.c 2>&1 | FileCheck %s

﻿// CHECK:      #ifdef TEST
// CHECK-NEXT: #include <string>
// CHECK-NEXT: #endif
