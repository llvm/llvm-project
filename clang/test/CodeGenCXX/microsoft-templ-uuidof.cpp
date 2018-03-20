// RUN: %clang_cc1 -emit-llvm %s -o - -DDEFINE_GUID -triple=i386-pc-win32 -fms-extensions | FileCheck %s

struct _GUID;

template <typename>
struct X {
};

struct __declspec(uuid("{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA}")) A {};

struct B {};

template <>
struct __declspec(uuid("{BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB}")) X<B> {};

struct __declspec(uuid("{CCCCCCCC-CCCC-CCCC-CCCC-CCCCCCCCCCCC}")) C {};

// CHECK-DAG: @_GUID_aaaaaaaa_aaaa_aaaa_aaaa_aaaaaaaaaaaa = linkonce_odr dso_local

const _GUID &xa = __uuidof(X<A>);
// CHECK-DAG:  @"?xa@@3ABU_GUID@@B" = {{.*}} @_GUID_aaaaaaaa_aaaa_aaaa_aaaa_aaaaaaaaaaaa

const _GUID &xb = __uuidof(X<B>);
// CHECK-DAG:  @"?xb@@3ABU_GUID@@B" = {{.*}} @_GUID_bbbbbbbb_bbbb_bbbb_bbbb_bbbbbbbbbbbb
const _GUID &xc = __uuidof(X<C>);
// CHECK-DAG:  @"?xc@@3ABU_GUID@@B" = {{.*}} @_GUID_cccccccc_cccc_cccc_cccc_cccccccccccc

template <>
struct __declspec(uuid("{DDDDDDDD-DDDD-DDDD-DDDD-DDDDDDDDDDDD}")) X<C> {};

template <typename>
struct __declspec(uuid("{EEEEEEEE-EEEE-EEEE-EEEE-EEEEEEEEEEEE}")) Y {
};

const _GUID &xd = __uuidof(X<C>);
// CHECK-DAG:  @"?xd@@3ABU_GUID@@B" = {{.*}} @_GUID_dddddddd_dddd_dddd_dddd_dddddddddddd

const _GUID &yd = __uuidof(Y<X<C> >);
// CHECK-DAG:  @"?yd@@3ABU_GUID@@B" = {{.*}} @_GUID_dddddddd_dddd_dddd_dddd_dddddddddddd
