// RUN: %clang_cc1 -fsycl-is-device -O0 -triple spirv64-unknown-unknown \
// RUN: -emit-llvm  %s -o - | FileCheck %s --check-prefix=DEVICE

// RUN: %clang_cc1 -fsycl-is-host -O0 -triple spirv64-unknown-unknown \
// RUN: -emit-llvm  %s -o - | FileCheck %s --check-prefix=HOST

// RUN: %clang_cc1 -fsycl-is-device -emit-llvm \
// RUN: -aux-triple x86_64-pc-windows-msvc -triple spir-unknown--unknown \
// RUN: %s -o - | FileCheck %s --check-prefix=MSVC

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
[[clang::sycl_kernel_entry_point(T)]] void f(T t) {
  t();
}

void g() {
  f(dg1);
  f(dg2);
  f(dg_inline1);
  f(dg_template<3>);
}

// HOST: @_ZN2QL3dg1E = internal global %class.anon undef, align 1
// HOST: @_ZN2QL3dg2E = internal global %class.anon.0 undef, align 1
// HOST: @_ZN2QL10dg_inline1E = linkonce_odr global %class.anon.2 undef, comdat, align 1
// HOST: @_ZN2QL11dg_templateILi3EEE = linkonce_odr global %class.anon.4 undef, comdat, align 1

// DEVICE: define spir_kernel void @_ZTSN2QL3dg1MUlvE_E
// DEVICE: call spir_func noundef i32 @_ZNK2QL3dg1MUlvE_clEv
// DEVICE: define internal spir_func noundef i32 @_ZNK2QL3dg1MUlvE_clEv
// DEVICE: define spir_kernel void @_ZTSN2QL3dg2MUlvE_E
// DEVICE: call spir_func noundef i32 @_ZNK2QL3dg2MUlvE_clEv
// DEVICE: define internal spir_func noundef i32 @_ZNK2QL3dg2MUlvE_clEv
// DEVICE: define spir_kernel void @_ZTSN2QL10dg_inline1MUlvE_E
// DEVICE: call spir_func noundef i32 @_ZNK2QL10dg_inline1MUlvE_clEv
// DEVICE: define linkonce_odr spir_func noundef i32 @_ZNK2QL10dg_inline1MUlvE_clEv
// DEVICE: define spir_kernel void @_ZTSN2QL11dg_templateILi3EEMUlvE_E
// DEVICE: call spir_func noundef i32 @_ZNK2QL11dg_templateILi3EEMUlvE_clEv
// DEVICE: define linkonce_odr spir_func noundef i32 @_ZNK2QL11dg_templateILi3EEMUlvE_clEv

// HOST: define spir_func void @_Z1gv
// HOST: call spir_func void @_Z1fIN2QL3dg1MUlvE_EEvT_
// HOST: call spir_func void @_Z1fIN2QL3dg2MUlvE_EEvT_
// HOST: call spir_func void @_Z1fIN2QL10dg_inline1MUlvE_EEvT_
// HOST: call spir_func void @_Z1fIN2QL11dg_templateILi3EEMUlvE_EEvT_
// HOST: define internal spir_func void @_Z1fIN2QL3dg1MUlvE_EEvT
// HOST: define internal spir_func void @_Z1fIN2QL3dg2MUlvE_EEvT_
// HOST: define linkonce_odr spir_func void @_Z1fIN2QL10dg_inline1MUlvE_EEvT_
// HOST: define linkonce_odr spir_func void @_Z1fIN2QL11dg_templateILi3EEMUlvE_EEvT_

// MSVC: define dso_local spir_kernel void @_ZTSN2QL3dg1MUlvE_E
// MSVC: call spir_func noundef i32 @_ZNK2QL3dg1MUlvE_clEv
// MSVC: define internal spir_func noundef i32 @_ZNK2QL3dg1MUlvE_clEv
// MSVC: define dso_local spir_kernel void @_ZTSN2QL3dg2MUlvE_E
// MSVC: call spir_func noundef i32 @_ZNK2QL3dg2MUlvE_clEv
// MSVC: define internal spir_func noundef i32 @_ZNK2QL3dg2MUlvE_clEv
// MSVC: define dso_local spir_kernel void @_ZTSN2QL10dg_inline1MUlvE_E
// MSVC: call spir_func noundef i32 @_ZNK2QL10dg_inline1MUlvE_clEv
// MSVC: define linkonce_odr spir_func noundef i32 @_ZNK2QL10dg_inline1MUlvE_clEv
// MSVC: define dso_local spir_kernel void @_ZTSN2QL11dg_templateILi3EEMUlvE_E
// MSVC: call spir_func noundef i32 @_ZNK2QL11dg_templateILi3EEMUlvE_clEv
// MSVC: define linkonce_odr spir_func noundef i32 @_ZNK2QL11dg_templateILi3EEMUlvE_clEv
