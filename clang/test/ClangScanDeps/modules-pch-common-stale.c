// Test that modifications to a common header (imported from both a PCH and a TU)
// cause rebuilds of dependent modules imported from the TU on incremental build.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- module.modulemap
module mod_common { header "mod_common.h" }
module mod_tu { header "mod_tu.h" }
module mod_tu_extra { header "mod_tu_extra.h" }

//--- mod_common.h
#define MOD_COMMON_MACRO 0

//--- mod_tu.h
#include "mod_common.h"
#if MOD_COMMON_MACRO
#include "mod_tu_extra.h"
#endif

//--- mod_tu_extra.h

//--- prefix.h
#include "mod_common.h"

//--- tu.c
#include "mod_tu.h"

// Clean: scan the PCH.
// RUN: clang-scan-deps -format experimental-full -o %t/deps_pch.json -- \
// RUN:     %clang -x c-header %t/prefix.h -o %t/prefix.h.pch -F %t \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache

// Clean: build the PCH.
// RUN: %deps-to-rsp %t/deps_pch.json --module-name mod_common > %t/mod_common.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/mod_common.rsp
// RUN: %clang @%t/pch.rsp

// Clean: scan the TU.
// RUN: clang-scan-deps -format experimental-full -o %t/deps_tu.json -- \
// RUN:     %clang -c %t/tu.c -o %t/tu.o -include %t/prefix.h -F %t \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache

// Clean: build the TU.
// RUN: %deps-to-rsp %t/deps_tu.json --module-name mod_tu > %t/mod_tu.rsp
// RUN: %deps-to-rsp %t/deps_tu.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/mod_tu.rsp
// RUN: %clang @%t/tu.rsp

// Incremental: modify the common module.
// RUN: echo "#define MOD_COMMON_MACRO 1" > %t/mod_common.h

// Incremental: scan the PCH.
// RUN: clang-scan-deps -format experimental-full -o %t/deps_pch.json -- \
// RUN:     %clang -x c-header %t/prefix.h -o %t/prefix.h.pch -F %t \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache

// Incremental: build the PCH.
// RUN: %deps-to-rsp %t/deps_pch.json --module-name mod_common > %t/mod_common.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/mod_common.rsp
// RUN: %clang @%t/pch.rsp

// Incremental: scan the TU. This needs to invalidate modules imported from the
//              TU that depend on modules imported from the PCH.
// RUN: clang-scan-deps -format experimental-full -o %t/deps_tu.json -- \
// RUN:     %clang -c %t/tu.c -o %t/tu.o -include %t/prefix.h -F %t \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache

// Incremental: build the TU.
// RUN: %deps-to-rsp %t/deps_tu.json --module-name mod_tu_extra > %t/mod_tu_extra.rsp
// RUN: %deps-to-rsp %t/deps_tu.json --module-name mod_tu > %t/mod_tu.rsp
// RUN: %deps-to-rsp %t/deps_tu.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/mod_tu_extra.rsp
// RUN: %clang @%t/mod_tu.rsp
// RUN: %clang @%t/tu.rsp
