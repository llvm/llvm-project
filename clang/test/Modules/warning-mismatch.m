// RUN: rm -rf %t.cache
// RUN: echo "@import Mismatch;" >%t.m
// RUN: %clang_cc1 -Wno-system-headers -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps \
// RUN:   -I%S/Inputs/warning-mismatch %t.m -fsyntax-only
// RUN: %clang_cc1 -Wsystem-headers -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps \
// RUN:   -I%S/Inputs/warning-mismatch %s -fsyntax-only

// This testcase triggers a warning flag mismatch in an already validated
// header.
@import Mismatch;
@import System;
