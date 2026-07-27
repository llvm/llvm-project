// // RUN: rm -rf %t
// // RUN: mkdir -p %t
//
// // Create PCH and ignore linker flags.
// // RUN: %clang -x c++-header %S/Inputs/pchfile.h -lm -o %t/pchfile.h.pch -### 2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE-LINK-FLAGS,CHECK-EMIT-PCH
// // RUN: %clang -x c++-header %S/Inputs/pchfile.h -lm -### 2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE-LINK-FLAGS,CHECK-EMIT-PCH
// // RUN: %clang --config %S/Inputs/config-l.cfg -x c++-header %S/Inputs/pchfile.h -o %t/pchfile.h.pch -### 2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE-LINK-FLAGS-CFG,CHECK-EMIT-PCH
//
// // CHECK-IGNORE-LINK-FLAGS: warning: -lm: 'linker' input unused
// // CHECK-IGNORE-LINK-FLAGS-NOT: clang: error: cannot specify -o when generating multiple output files
// // CHECK-EMIT-PCH: -emit-pch
// // CHECK-IGNORE-LINK-FLAGS-CFG: -Wall
// // CHECK-IGNORE-LINK-FLAGS-CFG-NOT: -lm --as-needed -Bstatic -lhappy -Bdynamic
