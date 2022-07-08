// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: c-index-test core -scan-deps %t -output-dir %t/out -- \
// RUN:   clang_tool -c %t/tu.c -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fimplicit-modules -fimplicit-module-maps \
// RUN:   -serialize-diagnostics %t/tu.diag -MD -MF %t/tu.d -o %t/tu.o \
// RUN: | FileCheck %s -DPREFIX=%/t -check-prefix=NONE

// NONE:        build-args:
// NONE-NOT:      -MT
// NONE-NOT:      -serialize-diagnostics-file
// NONE-NOT:      -dependency-file
// NONE:      build-args:
// NONE-SAME:   -fmodule-file=[[PREFIX]]/out/Mod_{{.*}}.pcm

// RUN: c-index-test core -scan-deps %t -output-dir %t/out -serialize-diagnostics -- \
// RUN:   clang_tool -c %t/tu.c -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fimplicit-modules -fimplicit-module-maps \
// RUN:   -serialize-diagnostics %t/tu.diag -MD -MF %t/tu.d -o %t/tu.o \
// RUN: | FileCheck %s -DPREFIX=%/t -check-prefix=DIAGS

// DIAGS:        build-args:
// DIAGS-NOT:      -MT
// DIAGS-NOT:      -dependency-file
// DIAGS-SAME:     -serialize-diagnostic-file [[PREFIX]]/out/Mod_{{.*}}.diag
// DIAGS-NOT:      -MT
// DIAGS-NOT:      -dependency-file
// DIAGS:      build-args:
// DIAGS-SAME:   -fmodule-file=[[PREFIX]]/out/Mod_{{.*}}.pcm

// RUN: c-index-test core -scan-deps %t -output-dir %t/out -dependency-file -- \
// RUN:   clang_tool -c %t/tu.c -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fimplicit-modules -fimplicit-module-maps \
// RUN:   -serialize-diagnostics %t/tu.diag -MD -MF %t/tu.d -o %t/tu.o \
// RUN: | FileCheck %s -DPREFIX=%/t -check-prefix=DEPS

// DEPS:        build-args:
// DEPS-NOT:      -serialize-diagnostic-file
// DEPS-SAME:     -MT [[PREFIX]]/out/Mod_{{.*}}.pcm
// DEPS-NOT:      -serialize-diagnostic-file
// DEPS-SAME:     -dependency-file [[PREFIX]]/out/Mod_{{.*}}.d
// DEPS-NOT:      -serialize-diagnostic-file
// DEPS:      build-args:
// DEPS-SAME:   -fmodule-file=[[PREFIX]]/out/Mod_{{.*}}.pcm

// RUN: c-index-test core -scan-deps %t -output-dir %t/out -dependency-file -dependency-target foo -- \
// RUN:   clang_tool -c %t/tu.c -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fimplicit-modules -fimplicit-module-maps \
// RUN:   -serialize-diagnostics %t/tu.diag -MD -MF %t/tu.d -o %t/tu.o \
// RUN: | FileCheck %s -DPREFIX=%/t -check-prefix=DEPS_MT1

// DEPS_MT1:        build-args:
// DEPS_MT1-NOT:      -serialize-diagnostic-file
// DEPS_MT1-SAME:     -MT foo
// DEPS_MT1-NOT:      -serialize-diagnostic-file
// DEPS_MT1:      build-args:
// DEPS_MT1-SAME:   -fmodule-file=[[PREFIX]]/out/Mod_{{.*}}.pcm

// RUN: c-index-test core -scan-deps %t -output-dir %t/out -dependency-file -dependency-target foo -dependency-target bar -- \
// RUN:   clang_tool -c %t/tu.c -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fimplicit-modules -fimplicit-module-maps \
// RUN:   -serialize-diagnostics %t/tu.diag -MD -MF %t/tu.d -o %t/tu.o \
// RUN: | FileCheck %s -DPREFIX=%/t -check-prefix=DEPS_MT2

// DEPS_MT2:        build-args:
// DEPS_MT2-NOT:      -serialize-diagnostic-file
// DEPS_MT2-SAME:     -MT foo
// DEPS_MT2-SAME:     -MT bar
// DEPS_MT2-NOT:      -serialize-diagnostic-file
// DEPS_MT2:      build-args:
// DEPS_MT2-SAME:   -fmodule-file=[[PREFIX]]/out/Mod_{{.*}}.pcm

// RUN: echo 'this_target_name_is_longer_than_the_256_byte_initial_buffer_size_to_test_that_we_alloc_and_call_again_with_a_sufficient_buffer_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_end' > %t/target-name.txt
// RUN: cat %t/target-name.txt > %t/long.txt
// RUN: c-index-test core -scan-deps %t -output-dir %t/out -dependency-file \
// RUN:     -dependency-target @%t/target-name.txt -- \
// RUN:   clang_tool -c %t/tu.c -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fimplicit-modules -fimplicit-module-maps \
// RUN:   -serialize-diagnostics %t/tu.diag -MD -MF %t/tu.d -o %t/tu.o \
// RUN: >> %t/long.txt
// RUN: FileCheck %s -check-prefix=LONG_OUT < %t/long.txt

// LONG_OUT: [[TARGET:this_target_.*_end]]
// LONG_OUT: -MT [[TARGET]]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h

//--- tu.c
#include "Mod.h"
