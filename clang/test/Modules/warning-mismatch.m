// RUN: rm -rf %t.cache
// RUN: echo "@import Mismatch;" >%t.m
// RUN: %clang_cc1 -Wno-system-headers -fdisable-module-hash -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps -I %S/Inputs/warning-mismatch %t.m -fsyntax-only -Rmodule-build
// RUN: echo -----------------------------------------------
// RUN: %clang_cc1 -Wsystem-headers -fdisable-module-hash -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps -I %S/Inputs/warning-mismatch %s -fsyntax-only -Rmodule-build

// This testcase triggers a warning flag mismatch in an already validated header.
@import Mismatch;
@import System;
