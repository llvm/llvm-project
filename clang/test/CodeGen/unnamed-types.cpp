// RUN: %clang_cc1 -O0 -triple x86_64-unknown-unknown \
// RUN: -emit-llvm  %s -o - | FileCheck %s

// RUN: %clang_cc1 -O0 -triple x86_64-pc-windows-msvc \
// RUN: -emit-llvm %s -o - | FileCheck %s --check-prefix=MSVC

namespace QL {
  auto dg1 = [] { return 1; };
  inline auto dg_inline1 = [] { return 1; };
}

namespace QL {
  auto dg2 = [] { return 2; };
  template<int N>
  auto dg_template = [] { return N; };
}

using namespace QL;
template<typename T>
void f(T t) {
  t();
}

void g() {
  f(dg1);
  f(dg2);
  f(dg_inline1);
  f(dg_template<3>);
}

// CHECK: @_ZN2QL3dg1E = internal global %class.anon undef, align 1
// CHECK: @_ZN2QL3dg2E = internal global %class.anon.0 undef, align 1
// CHECK: @_ZN2QL10dg_inline1E = linkonce_odr global %class.anon.2 undef, comdat, align 1
// CHECK: @_ZN2QL11dg_templateILi3EEE = linkonce_odr global %class.anon.4 undef, comdat, align 1

// MSVC: @"?dg1@QL@@3V<lambda_0>@1@A" = internal global %class.anon undef, align 1
// MSVC: @"?dg2@QL@@3V<lambda_1>@1@A" = internal global %class.anon.0 undef, align 1
// MSVC: @"?dg_inline1@QL@@3V<lambda_1>@01@A" = linkonce_odr dso_local global %class.anon.2 undef, comdat, align 1
// MSVC: @"??$dg_template@$02@QL@@3V<lambda_1>@01@A" = linkonce_odr dso_local global %class.anon.4 undef, comdat, align 1


// CHECK: define internal void @"_Z1fIN2QL3$_0EEvT_"
// CHECK: call noundef i32 @"_ZNK2QL3$_0clEv"
// CHECK: define internal void @"_Z1fIN2QL3$_1EEvT_"
// CHECK: define linkonce_odr void @_Z1fIN2QL10dg_inline1MUlvE_EEvT_
// CHECK: call noundef i32 @_ZNK2QL10dg_inline1MUlvE_clEv
// CHECK: define linkonce_odr void @_Z1fIN2QL11dg_templateILi3EEMUlvE_EEvT_
// CHECK: call noundef i32 @_ZNK2QL11dg_templateILi3EEMUlvE_clEv
// CHECK: define internal noundef i32 @"_ZNK2QL3$_0clEv"
// CHECK: define internal noundef i32 @"_ZNK2QL3$_1clEv"
// CHECK: define linkonce_odr noundef i32 @_ZNK2QL10dg_inline1MUlvE_clEv
// CHECK: define linkonce_odr noundef i32 @_ZNK2QL11dg_templateILi3EEMUlvE_clEv

// MSVC: define linkonce_odr dso_local void @"??$f@V<lambda_1>@dg_inline1@QL@@@@YAXV<lambda_1>@dg_inline1@QL@@@Z"
// MSVC: call noundef i32 @"??R<lambda_1>@dg_inline1@QL@@QEBA?A?<auto>@@XZ"
// MSVC: define linkonce_odr dso_local void @"??$f@V<lambda_1>@?$dg_template@$02@QL@@@@YAXV<lambda_1>@?$dg_template@$02@QL@@@Z"
// MSVC: call noundef i32 @"??R<lambda_1>@?$dg_template@$02@QL@@QEBA?A?<auto>@@XZ"
// MSVC: define internal noundef i32 @"??R<lambda_0>@QL@@QEBA?A?<auto>@@XZ"
// MSVC: define internal noundef i32 @"??R<lambda_1>@QL@@QEBA?A?<auto>@@XZ"
// MSVC: define linkonce_odr dso_local noundef i32 @"??R<lambda_1>@dg_inline1@QL@@QEBA?A?<auto>@@XZ"
// MSVC: define linkonce_odr dso_local noundef i32 @"??R<lambda_1>@?$dg_template@$02@QL@@QEBA?A?<auto>@@XZ"
