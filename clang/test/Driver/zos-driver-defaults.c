// RUN: %clang -### --target=s390x-none-zos -fsyntax-only %s 2>&1 | FileCheck --check-prefixes=CHECK-C-MACRO,CHECK-SHORT-ENUMS %s
// RUN: %clang -### --target=s390x-none-zos -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-ZOS-INCLUDES %s
// RUN: %clang -### --target=s390x-none-zos -fno-short-enums -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clangxx -### --target=s390x-none-zos -fsyntax-only %s 2>&1 | FileCheck --check-prefixes=CHECK-C-MACRO,CHECK-CXX-MACRO %s
// RUN: %clang -### --target=s390x-none-zos -x c++ -fsyntax-only %s 2>&1 | FileCheck --check-prefixes=CHECK-C-MACRO,CHECK-CXX-MACRO %s

//CHECK-C-MACRO: -D_UNIX03_WITHDRAWN
//CHECK-C-MACRO: -D_OPEN_DEFAULT

//CHECK-CXX-MACRO: -D_XOPEN_SOURCE=600
//CHECK-USER-CXX-MACRO-NOT: -D_XOPEN_SOURCE=600
//CHECK-USER-CXX-MACRO: "-D" "_XOPEN_SOURCE=700"

//CHECK-SHORT-ENUMS: -fshort-enums
//CHECK-SHORT-ENUMS: -fno-signed-char

//CHECK-ZOS-INCLUDES: clang{{.*}} "-cc1" "-triple" "s390x-none-zos"
//CHECK-ZOS-INCLUDES-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
//CHECK-ZOS-INCLUDES-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
//CHECK-ZOS-INCLUDES-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
//
//CHECK-NOT: -fshort-enums
//CHECK: -fno-signed-char
