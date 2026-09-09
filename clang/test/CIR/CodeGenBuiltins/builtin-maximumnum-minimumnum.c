// RUN: %clang_cc1 -x c++ -std=c++20 -disable-llvm-passes -O3 -triple x86_64 %s -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -x c++ -std=c++20 -disable-llvm-passes -O3 -triple x86_64 %s -emit-llvm -fclangir -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll --check-prefix=LLVM
// RUN: %clang_cc1 -x c++ -std=c++20 -disable-llvm-passes -O3 -triple x86_64 %s -emit-llvm -o %t-ogcg.ll
// RUN: FileCheck %s --input-file=%t-ogcg.ll --check-prefix=LLVM

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef long double ldouble2 __attribute__((ext_vector_type(2)));

// CIR-LABEL: @_Z7pfmin16Dv8_DF16_S_(
// CIR:          cir.call_llvm_intrinsic "minimumnum" %{{.*}}, %{{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>) -> !cir.vector<8 x !cir.f16>
//
// LLVM-LABEL: @_Z7pfmin16Dv8_DF16_S_(
// LLVM:         call <8 x half> @llvm.minimumnum.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}})
//
half8 pfmin16(half8 a, half8 b) {
	return __builtin_elementwise_minimumnum(a, b);
}
// CIR-LABEL: @_Z8pfmin16bDv8_DF16bS_(
// CIR:          cir.call_llvm_intrinsic "minimumnum" %{{.*}}, %{{.*}} : (!cir.vector<8 x !cir.bf16>, !cir.vector<8 x !cir.bf16>) -> !cir.vector<8 x !cir.bf16>
//
// LLVM-LABEL: @_Z8pfmin16bDv8_DF16bS_(
// LLVM:         call <8 x bfloat> @llvm.minimumnum.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
//
bf16x8 pfmin16b(bf16x8 a, bf16x8 b) {
	return __builtin_elementwise_minimumnum(a, b);
}
// CIR-LABEL: @_Z7pfmin32Dv4_fS_(
// CIR:          cir.call_llvm_intrinsic "minimumnum" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>
//
// LLVM-LABEL: @_Z7pfmin32Dv4_fS_(
// LLVM:         call <4 x float> @llvm.minimumnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
//
float4 pfmin32(float4 a, float4 b) {
	return __builtin_elementwise_minimumnum(a, b);
}
// CIR-LABEL: @_Z7pfmin64Dv2_dS_(
// CIR:          cir.call_llvm_intrinsic "minimumnum" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>
//
// LLVM-LABEL: @_Z7pfmin64Dv2_dS_(
// LLVM:         call <2 x double> @llvm.minimumnum.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}})
//
double2 pfmin64(double2 a, double2 b) {
	return __builtin_elementwise_minimumnum(a, b);
}
// CIR-LABEL: @_Z7pfmin80v(
// CIR:          cir.call_llvm_intrinsic "minimumnum" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.long_double<!cir.f80>>, !cir.vector<2 x !cir.long_double<!cir.f80>>) -> !cir.vector<2 x !cir.long_double<!cir.f80>>
//
// LLVM-LABEL: @_Z7pfmin80v(
// LLVM:         call <2 x x86_fp80> @llvm.minimumnum.v2f80(<2 x x86_fp80> %{{.*}}, <2 x x86_fp80> %{{.*}})
//
void pfmin80() {
  ldouble2 a, b;
	ldouble2 c = __builtin_elementwise_minimumnum(a, b);
}

// CIR-LABEL: @_Z7pfmax16Dv8_DF16_S_(
// CIR:          cir.call_llvm_intrinsic "maximumnum" %{{.*}}, %{{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>) -> !cir.vector<8 x !cir.f16>
//
// LLVM-LABEL: @_Z7pfmax16Dv8_DF16_S_(
// LLVM:         call <8 x half> @llvm.maximumnum.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}})
//
half8 pfmax16(half8 a, half8 b) {
	return __builtin_elementwise_maximumnum(a, b);
}
// CIR-LABEL: @_Z8pfmax16bDv8_DF16bS_(
// CIR:          cir.call_llvm_intrinsic "maximumnum" %{{.*}}, %{{.*}} : (!cir.vector<8 x !cir.bf16>, !cir.vector<8 x !cir.bf16>) -> !cir.vector<8 x !cir.bf16>
//
// LLVM-LABEL: @_Z8pfmax16bDv8_DF16bS_(
// LLVM:         call <8 x bfloat> @llvm.maximumnum.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
//
bf16x8 pfmax16b(bf16x8 a, bf16x8 b) {
	return __builtin_elementwise_maximumnum(a, b);
}
// CIR-LABEL: @_Z7pfmax32Dv4_fS_(
// CIR:          cir.call_llvm_intrinsic "maximumnum" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float> loc(#loc93)
//
// LLVM-LABEL: @_Z7pfmax32Dv4_fS_(
// LLVM:         call <4 x float> @llvm.maximumnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
//
float4 pfmax32(float4 a, float4 b) {
	return __builtin_elementwise_maximumnum(a, b);
}
// CIR-LABEL: @_Z7pfmax64Dv2_dS_(
// CIR:          cir.call_llvm_intrinsic "maximumnum" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>
//
// LLVM-LABEL: @_Z7pfmax64Dv2_dS_(
// LLVM:         call <2 x double> @llvm.maximumnum.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}})
//
double2 pfmax64(double2 a, double2 b) {
	return __builtin_elementwise_maximumnum(a, b);
}

// CIR-LABEL: @_Z7pfmax80v(
// CIR:          cir.call_llvm_intrinsic "minimumnum" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.long_double<!cir.f80>>, !cir.vector<2 x !cir.long_double<!cir.f80>>) -> !cir.vector<2 x !cir.long_double<!cir.f80>>
//
// LLVM-LABEL: @_Z7pfmax80v(
// LLVM:         call <2 x x86_fp80> @llvm.minimumnum.v2f80(<2 x x86_fp80> %{{.*}}, <2 x x86_fp80> %{{.*}})
//
void pfmax80() {
  ldouble2 a, b;
	ldouble2 c = __builtin_elementwise_minimumnum(a, b);
}
