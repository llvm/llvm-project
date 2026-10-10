// RUN: split-file %s %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/M.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=RAW,O0
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -DINCLUDE_FIRST %t/use.cpp -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=RAW,O0
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -O2 %t/use.cpp -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=RAW,O2
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -O2 -DINCLUDE_FIRST %t/use.cpp -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=RAW,O2
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -emit-llvm -o - | FileCheck %s --check-prefix=INLINE --implicit-check-not=always_wrapper
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -DINCLUDE_FIRST %t/use.cpp -emit-llvm -o - | FileCheck %s --check-prefix=INLINE --implicit-check-not=always_wrapper
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -O2 %t/use.cpp -emit-llvm -o - | FileCheck %s --check-prefix=INLINE --implicit-check-not=always_wrapper
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -O2 -DINCLUDE_FIRST %t/use.cpp -emit-llvm -o - | FileCheck %s --check-prefix=INLINE --implicit-check-not=always_wrapper
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t %t/address.cpp -emit-llvm -o - | FileCheck %s --check-prefix=ADDRESS
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -DINCLUDE_FIRST %t/address.cpp -emit-llvm -o - | FileCheck %s --check-prefix=ADDRESS
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -O2 %t/address.cpp -emit-llvm -o - | FileCheck %s --check-prefix=ADDRESS
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fprebuilt-module-path=%t -O2 -DINCLUDE_FIRST %t/address.cpp -emit-llvm -o - | FileCheck %s --check-prefix=ADDRESS

// RAW-DAG: define available_externally{{.*}} @always_wrapper(
// RAW-DAG: attributes #{{[0-9]+}} = { alwaysinline
// RAW-DAG: declare{{.*}} @{{.*}}named_wrapper{{.*}}(
// O0-DAG: declare{{.*}} @plain_wrapper(
// O2-DAG: define available_externally{{.*}} @plain_wrapper(
// INLINE-LABEL: define{{.*}} @test_always(
// INLINE: call{{.*}} @real_fn(
// ADDRESS-LABEL: define{{.*}} @address_always(
// ADDRESS: ret ptr @always_wrapper
// ADDRESS-LABEL: define{{.*}} @address_plain(
// ADDRESS: ret ptr @plain_wrapper

//--- wrappers.h
extern "C" {
long real_fn(long);

extern inline __attribute__((gnu_inline, always_inline))
long always_wrapper(long value) { return real_fn(value); }

extern inline __attribute__((gnu_inline))
long plain_wrapper(long value) { return real_fn(value); }
}

//--- M.cppm
module;
#include "wrappers.h"
export module M;

export extern inline __attribute__((gnu_inline, always_inline))
long named_wrapper(long value) { return value + 1; }

//--- use.cpp
#ifdef INCLUDE_FIRST
#include "wrappers.h"
import M;
#else
import M;
#include "wrappers.h"
#endif

extern "C" long test_always() { return always_wrapper(41); }
extern "C" long test_plain() { return plain_wrapper(42); }
extern "C" long test_named() { return named_wrapper(43); }

//--- address.cpp
#ifdef INCLUDE_FIRST
#include "wrappers.h"
import M;
#else
import M;
#include "wrappers.h"
#endif

extern "C" auto address_always() -> long (*)(long) {
	return &always_wrapper;
}

extern "C" auto address_plain() -> long (*)(long) {
	return &plain_wrapper;
}