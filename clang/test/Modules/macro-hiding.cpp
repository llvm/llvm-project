// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB1 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB1 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB1 -DB2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB1 -DB2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB1 -DB2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DB1 -DB2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB1 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB1 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB1 -DB2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB1 -DB2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB1 -DB2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA2 -DB1 -DB2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB1 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB1 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB1 -DB2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB1 -DB2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB1 -DB2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DB1 -DB2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB2 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB1 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB1 -DC1 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB1 -DB2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB1 -DB2 -DD1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB1 -DB2 -DC1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DA1 -DA2 -DB1 -DB2 -DC1 -DD1
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/macro-hiding %s -DE1

#ifdef A1
#include "a1.h"
#endif

#ifdef A2
#include "a2.h"
#endif

#ifdef B1
#include "b1.h"
#endif

#ifdef B2
#include "b2.h"
#endif

#ifdef C1
#include "c1.h"
#endif

#ifdef D1
#include "d1.h"
#endif

#ifdef E1
#include "e1.h"
#endif

#ifdef E2
#include "e2.h"
#endif

#if defined(A1) || defined(B2) || defined(C1) || defined(D1) || defined(E1) || defined(E2)
void h() { assert(true); }
#else
void assert() {}
#endif
