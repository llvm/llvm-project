// RUN: rm -rf %t
// RUN: split-file %s %t

//--- module.modulemap
module A { header "A.h" }
module B { header "B.h" }
//--- A.h
//--- B.h
//--- pch.h
#include "A.h"
//--- tu.c
#include "B.h"

// RUN: clang-scan-deps -format experimental-full -module-files-dir %t/build -o %t/result_pch.json \
// RUN:   -- %clang -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:      -x c-header %t/pch.h -o %t/pch.h.pch \
// RUN:      -fapplication-extension
// RUN: %deps-to-rsp %t/result_pch.json --module-name=A > %t/A.rsp
// RUN: %deps-to-rsp %t/result_pch.json --tu-index=0 > %t/pch.rsp
// RUN: %clang @%t/A.rsp
// RUN: %clang @%t/pch.rsp

// RUN: clang-scan-deps -format experimental-full -module-files-dir %t/build -o %t/result_tu.json \
// RUN:   -- %clang -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:      -c %t/tu.c -o %t/tu.o -include %t/pch.h \
// RUN:      -fapplication-extension
// RUN: %deps-to-rsp %t/result_tu.json --module-name=B > %t/B.rsp
// RUN: %deps-to-rsp %t/result_tu.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/B.rsp
// RUN: %clang @%t/tu.rsp
