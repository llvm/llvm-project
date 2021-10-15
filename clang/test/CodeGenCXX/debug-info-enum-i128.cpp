// RUN: %clang_cc1 %s -triple x86_64-windows-msvc -gcodeview -debug-info-kind=limited -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -debug-info-kind=limited -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -debug-info-kind=limited -emit-llvm -o - | FileCheck %s

// FIXME: llvm.org/pr51221, the APSInt leaks
// UNSUPPORTED: asan

enum class uns : __uint128_t { unsval = __uint128_t(1) << 64 };
uns t1() { return uns::unsval; }

enum class sig : __int128 { sigval = -(__int128(1) << 64) };
sig t2() { return sig::sigval; }


// CHECK-LABEL: !DICompositeType(tag: DW_TAG_enumeration_type, name: "uns", {{.*}})
// CHECK: !DIEnumerator(name: "unsval", value: 18446744073709551616, isUnsigned: true)

// CHECK-LABEL: !DICompositeType(tag: DW_TAG_enumeration_type, name: "sig", {{.*}})
// CHECK: !DIEnumerator(name: "sigval", value: -18446744073709551616)
