// RUN: rm -rf %t
// RUN: mkdir -p %t/prebuilt_modules
//
// RUN: %clang_cc1 -triple %itanium_abi_triple                          \
// RUN:     -fmodules-ts -fprebuilt-module-path=%t/prebuilt-modules     \
// RUN:     -emit-module-interface -pthread -DBUILD_MODULE              \
// RUN:     %s -o %t/prebuilt_modules/mismatching_module.pcm
//
// RUN: not %clang_cc1 -triple %itanium_abi_triple -fmodules-ts         \
// RUN:     -fprebuilt-module-path=%t/prebuilt_modules -DCHECK_MISMATCH \
// RUN:     %s 2>&1 | FileCheck %s

#ifdef BUILD_MODULE
export module mismatching_module;
#endif

#ifdef CHECK_MISMATCH
import mismatching_module;
// CHECK: error: POSIX thread support was enabled in PCH file but is currently disabled
// CHECK-NEXT: module file {{.*[/|\\\\]}}mismatching_module.pcm cannot be loaded due to a configuration mismatch with the current compilation
#endif

