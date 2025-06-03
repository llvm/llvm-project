// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: mkdir -p %t/prebuilt_modules
//
// RUN: %clang_cc1 -triple %itanium_abi_triple                           \
// RUN:     -std=c++20 -fprebuilt-module-path=%t/prebuilt-modules        \
// RUN:     -emit-module-interface -pthread -DBUILD_MODULE               \
// RUN:     %t/mismatching_module.cppm -o                                \
// RUN:     %t/prebuilt_modules/mismatching_module.pcm
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20                \
// RUN:     -fprebuilt-module-path=%t/prebuilt_modules -DCHECK_MISMATCH  \
// RUN:     %t/use.cpp 2>&1 | FileCheck %t/use.cpp

// Test again with reduced BMI.
// RUN: %clang_cc1 -triple %itanium_abi_triple                           \
// RUN:     -std=c++20 -fprebuilt-module-path=%t/prebuilt-modules        \
// RUN:     -emit-reduced-module-interface -pthread -DBUILD_MODULE       \
// RUN:     %t/mismatching_module.cppm -o                                \
// RUN:     %t/prebuilt_modules/mismatching_module.pcm
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20                \
// RUN:     -fprebuilt-module-path=%t/prebuilt_modules -DCHECK_MISMATCH  \
// RUN:     %t/use.cpp 2>&1 | FileCheck %t/use.cpp
//
// RUN: %clang_cc1 -triple %itanium_abi_triple                           \
// RUN:     -std=c++20 -fprebuilt-module-path=%t/prebuilt-modules        \
// RUN:     -emit-module-interface -pthread -DBUILD_MODULE               \
// RUN:     %t/mismatching_module.cppm -o                                \
// RUN:     %t/prebuilt_modules/mismatching_module.pcm
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20                \
// RUN:     -fprebuilt-module-path=%t/prebuilt_modules -DCHECK_MISMATCH  \
// RUN:     -Wno-module-mismatched-option %t/use.cpp 2>&1 | FileCheck %t/use.cpp \
// RUN:     --check-prefix=NOWARN --allow-empty

// Test again with reduced BMI.
// RUN: %clang_cc1 -triple %itanium_abi_triple                           \
// RUN:     -std=c++20 -fprebuilt-module-path=%t/prebuilt-modules        \
// RUN:     -emit-reduced-module-interface -pthread -DBUILD_MODULE       \
// RUN:     %t/mismatching_module.cppm -o                                \
// RUN:     %t/prebuilt_modules/mismatching_module.pcm
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20                \
// RUN:     -fprebuilt-module-path=%t/prebuilt_modules -DCHECK_MISMATCH  \
// RUN:     -Wno-module-mismatched-option %t/use.cpp 2>&1 | FileCheck %t/use.cpp \
// RUN:     --check-prefix=NOWARN --allow-empty

//--- mismatching_module.cppm
export module mismatching_module;

//--- use.cpp
import mismatching_module;
// CHECK: warning: POSIX thread support was enabled in AST file '{{.*[/|\\\\]}}mismatching_module.pcm' but is currently disabled

// NOWARN-NOT: warning
