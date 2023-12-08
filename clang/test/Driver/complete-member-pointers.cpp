// RUN: %clangxx -### -c %s -o %t.o -target x86_64-unknown-linux 2>&1 | FileCheck --check-prefix=NOFLAG %s
// RUN: %clangxx -### -c %s -o %t.o -target x86_64-unknown-linux -fcomplete-member-pointers 2>&1 | FileCheck %s
// RUN: %clangxx -### -c %s -o %t.o -target x86_64-unknown-linux -fcomplete-member-pointers -fno-complete-member-pointers 2>&1 | FileCheck --check-prefix=NOFLAG %s
// RUN: %clang_cl -### /c /Fo%t.o -target x86_64-pc-win32 -fcomplete-member-pointers -- %s 2>&1 | FileCheck %s

// CHECK: "-fcomplete-member-pointers"
// NOFLAG-NOT: "-fcomplete-member-pointers"
