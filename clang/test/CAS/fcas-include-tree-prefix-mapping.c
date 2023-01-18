// RUN: rm -rf %t
// RUN: split-file %s %t/src
// RUN: mkdir %t/out

// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree \
// RUN:   -fdepscan-prefix-map=%t/src=/^src -fdepscan-prefix-map=%t/out=/^out -fdepscan-prefix-map-toolchain=/^toolchain -fdepscan-prefix-map-sdk=/^sdk \
// RUN:   -o %t/t1.rsp -cc1-args \
// RUN:     -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache -Rcompile-job-cache \
// RUN:       -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:       -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:       -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb -fdebug-compilation-dir=%t/out \
// RUN:       -emit-llvm %t/src/main.c -o %t/out/output.ll -include %t/src/prefix.h -I %t/src/inc \
// RUN:       -MT deps -dependency-file %t/t1.d
// RUN: %clang @%t/t1.rsp 2> %t/output.txt

// RUN: cat %t/output.txt | sed \
// RUN:   -e "s/^.*miss for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key

// RUN: clang-cas-test -print-compile-job-cache-key -cas %t/cas @%t/cache-key > %t/printed-key.txt
// RUN: FileCheck %s -input-file %t/printed-key.txt -DSRC_PREFIX=%t/src -DOUT_PREFIX=%t/out -DSDK_PREFIX=%S/Inputs/SDK -DTOOLCHAIN_PREFIX=%S/Inputs/toolchain_dir

// CHECK-NOT: [[SRC_PREFIX]]
// CHECK-NOT: [[OUT_PREFIX]]
// CHECK-NOT: [[SDK_PREFIX]]
// CHECK-NOT: [[TOOLCHAIN_PREFIX]]
// CHECK: /^src{{[/\\]}}main.c
// CHECK: /^src{{[/\\]}}inc{{[/\\]}}t.h
// CHECK: /^toolchain{{[/\\]}}lib{{[/\\]}}clang{{[/\\]}}1000{{[/\\]}}include{{[/\\]}}stdarg.h
// CHECK: /^sdk{{[/\\]}}usr{{[/\\]}}include{{[/\\]}}stdlib.h

// RUN: FileCheck %s -input-file %t/out/output.ll -check-prefix=IR -DSRC_PREFIX=%t/src -DOUT_PREFIX=%t/out -DSDK_PREFIX=%S/Inputs/SDK -DTOOLCHAIN_PREFIX=%S/Inputs/toolchain_dir
// IR-NOT: [[SRC_PREFIX]]
// IR-NOT: [[OUT_PREFIX]]
// IR-NOT: [[SDK_PREFIX]]
// IR-NOT: [[TOOLCHAIN_PREFIX]]

// Check with prefix header.

// RUN: split-file %s %t/src2
// RUN: mkdir %t/out2

// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree \
// RUN:   -fdepscan-prefix-map=%t/src2=/^src -fdepscan-prefix-map=%t/out2=/^out -fdepscan-prefix-map-toolchain=/^toolchain -fdepscan-prefix-map-sdk=/^sdk \
// RUN:   -o %t/t2.rsp -cc1-args \
// RUN:     -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache -Rcompile-job-cache \
// RUN:       -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:       -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:       -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb -fdebug-compilation-dir=%t/out2 \
// RUN:       -emit-llvm %t/src2/main.c -o %t/out2/output.ll -include %t/src2/prefix.h -I %t/src2/inc \
// RUN:       -MT deps -dependency-file %t/t2.d
// RUN: %clang @%t/t2.rsp 2> %t/output2.txt

// RUN: cat %t/output2.txt | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key2

// RUN: diff -u %t/cache-key %t/cache-key2

// Check dependencies.

// Baseline for comparison.
// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:   -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:   -emit-obj %t/src/main.c -o %t/out/main.o -include %t/src/prefix.h -I %t/src/inc \
// RUN:   -MT deps -dependency-file %t/regular1.d
// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:   -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:   -emit-obj %t/src2/main.c -o %t/out2/main.o -include %t/src2/prefix.h -I %t/src2/inc \
// RUN:   -MT deps -dependency-file %t/regular2.d

// RUN: diff -u %t/regular1.d %t/t1.d
// RUN: diff -u %t/regular2.d %t/t2.d

// Check with PCH.

// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree \
// RUN:   -fdepscan-prefix-map=%t/src=/^src -fdepscan-prefix-map=%t/out=/^out -fdepscan-prefix-map-toolchain=/^toolchain -fdepscan-prefix-map-sdk=/^sdk \
// RUN:   -o %t/pch1.rsp -cc1-args \
// RUN:     -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache -Rcompile-job-cache \
// RUN:       -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:       -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:       -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb -fdebug-compilation-dir=%t/out \
// RUN:       -emit-pch -x c-header %t/src/prefix.h -o %t/out/prefix.h.pch -include %t/src/prefix.h -I %t/src/inc
// RUN: %clang @%t/pch1.rsp

// With different cas path to avoid cache hit.
// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree \
// RUN:   -fdepscan-prefix-map=%t/src2=/^src -fdepscan-prefix-map=%t/out2=/^out -fdepscan-prefix-map-toolchain=/^toolchain -fdepscan-prefix-map-sdk=/^sdk \
// RUN:   -o %t/pch2.rsp -cc1-args \
// RUN:     -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas2 -faction-cache-path %t/cache2 -Rcompile-job-cache \
// RUN:       -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:       -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:       -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb -fdebug-compilation-dir=%t/out2 \
// RUN:       -emit-pch -x c-header %t/src2/prefix.h -o %t/out2/prefix.h.pch -include %t/src2/prefix.h -I %t/src2/inc
// RUN: %clang @%t/pch2.rsp

// RUN: diff %t/out/prefix.h.pch %t/out2/prefix.h.pch

// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree \
// RUN:   -fdepscan-prefix-map=%t/src=/^src -fdepscan-prefix-map=%t/out=/^out -fdepscan-prefix-map-toolchain=/^toolchain -fdepscan-prefix-map-sdk=/^sdk \
// RUN:   -o %t/t3.rsp -cc1-args \
// RUN:     -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache -Rcompile-job-cache \
// RUN:       -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:       -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:       -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb -fdebug-compilation-dir=%t/out \
// RUN:       -emit-obj %t/src/main.c -o %t/out/main.o -include-pch %t/out/prefix.h.pch -I %t/src/inc \
// RUN:       -MT deps -dependency-file %t/t1.pch.d
// RUN: %clang @%t/t3.rsp 2> %t/output3.txt

// RUN: cat %t/output3.txt | sed \
// RUN:   -e "s/^.*miss for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key3

// RUN: %clang -cc1depscan -fdepscan=inline -fdepscan-include-tree \
// RUN:   -fdepscan-prefix-map=%t/src2=/^src -fdepscan-prefix-map=%t/out2=/^out -fdepscan-prefix-map-toolchain=/^toolchain -fdepscan-prefix-map-sdk=/^sdk \
// RUN:   -o %t/t4.rsp -cc1-args \
// RUN:     -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache -Rcompile-job-cache \
// RUN:       -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:       -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:       -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb -fdebug-compilation-dir=%t/out2 \
// RUN:       -emit-obj %t/src2/main.c -o %t/out2/main.o -include-pch %t/out2/prefix.h.pch -I %t/src2/inc \
// RUN:       -MT deps -dependency-file %t/t2.pch.d
// RUN: %clang @%t/t4.rsp 2> %t/output4.txt

// RUN: cat %t/output4.txt | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key4

// RUN: diff -u %t/cache-key3 %t/cache-key4
// RUN: diff %t/out/main.o %t/out2/main.o

// Check dependencies.

// Baseline for comparison.
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache -Rcompile-job-cache \
// RUN:   -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:   -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:   -emit-pch -x c-header %t/src/prefix.h -o %t/out/reg-prefix.h.pch -include %t/src/prefix.h -I %t/src/inc
// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:   -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:   -emit-obj %t/src/main.c -o %t/out/main.o -include-pch %t/out/reg-prefix.h.pch -I %t/src/inc \
// RUN:   -MT deps -dependency-file %t/regular1.pch.d
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache -Rcompile-job-cache \
// RUN:   -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:   -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:   -emit-pch -x c-header %t/src2/prefix.h -o %t/out2/reg-prefix.h.pch -include %t/src2/prefix.h -I %t/src2/inc
// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -internal-isystem %S/Inputs/toolchain_dir/lib/clang/1000/include \
// RUN:   -isysroot %S/Inputs/SDK -internal-externc-isystem %S/Inputs/SDK/usr/include \
// RUN:   -emit-obj %t/src2/main.c -o %t/out2/main.o -include-pch %t/out2/reg-prefix.h.pch -I %t/src2/inc \
// RUN:   -MT deps -dependency-file %t/regular2.pch.d

// RUN: diff -u %t/regular1.pch.d %t/t1.pch.d
// RUN: diff -u %t/regular2.pch.d %t/t2.pch.d

//--- main.c
#include "t.h"
#include <stdarg.h>
#include <stdlib.h>

int test(void) {
  return SOME_VALUE;
}

//--- inc/t.h

//--- prefix.h
#define SOME_VALUE 3
#include "pt.h"

//--- inc/pt.h
#include "../inc2/pt2.h"

//--- inc2/pt2.h
