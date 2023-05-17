// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: mkdir -p %t/prebuilt_modules
//
// RUN: %clang_cc1 -triple %itanium_abi_triple                          \
// RUN:     -std=c++20 -fprebuilt-module-path=%t/prebuilt-modules       \
// RUN:     -emit-module-interface -pthread -DBUILD_MODULE              \
// RUN:     %t/mismatching_module.cppm -o                               \
// RUN:     %t/prebuilt_modules/mismatching_module.pcm
//
// RUN: not %clang_cc1 -triple %itanium_abi_triple -std=c++20           \
// RUN:     -fprebuilt-module-path=%t/prebuilt_modules -DCHECK_MISMATCH \
// RUN:     %t/use.cpp 2>&1 | FileCheck %s

//--- mismatching_module.cppm
export module mismatching_module;

//--- use.cpp
import mismatching_module;
// CHECK: error: POSIX thread support was enabled in PCH file but is currently disabled
// CHECK-NEXT: module file {{.*[/|\\\\]}}mismatching_module.pcm cannot be loaded due to a configuration mismatch with the current compilation
