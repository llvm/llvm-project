// Test UTF8 BOM at start of file
// RUN: printf '\357\273\277' > %t.c
﻿// RUN: echo '#ifdef TEST\n' >> %t.c
// RUN: echo '#include <string>' >> %t.c
// RUN: echo '#endif' >> %t.c
// RUN: %clang_cc1 -DTEST -print-dependency-directives-minimized-source %t.c 2>&1 | FileCheck %s

﻿// CHECK:      #ifdef TEST
// CHECK-NEXT: #include <string>
// CHECK-NEXT: #endif
