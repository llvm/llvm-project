// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t/t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 %t/main.c -emit-obj -o %t/output.o -isystem %t/sys \
// RUN:     -MT deps -dependency-file %t/t.d
// RUN: FileCheck %s -input-file=%t/t.d -check-prefix=NOSYS
// RUN: FileCheck %s -input-file=%t/t.d -check-prefix=COMMON

// Including system headers.
// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree -o %t/t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 %t/main.c -emit-obj -o %t/output.o -isystem %t/sys \
// RUN:     -MT deps -sys-header-deps -dependency-file %t/t-sys.d
// RUN: FileCheck %s -input-file=%t/t-sys.d -check-prefix=WITHSYS -check-prefix=COMMON

// NOSYS-NOT: sys.h
// COMMON: main.c
// COMMON: my_header.h
// WITHSYS: sys.h

//--- main.c
#include "my_header.h"
#include <sys.h>

//--- my_header.h

//--- sys/sys.h
