// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC
// RUN: %clang_cc1 -mfloat-abi hard -triple armv7-unknown-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM32
// RUN: %clang_cc1 -mfloat-abi hard -triple aarch64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM64
// RUN: %clang_cc1 -mfloat-abi hard -triple x86_64-unknown-windows-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=X64

#if defined(__x86_64__)
#define CC __attribute__((vectorcall))
#else
#define CC
#endif

// Test that C++ classes are correctly classified as homogeneous aggregates.

struct Base1 {
  int x;
};
struct Base2 {
  double x;
};
struct Base3 {
  double x;
};
struct D1 : Base1 {  // non-homogeneous aggregate
  double y, z;
};
struct D2 : Base2 {  // homogeneous aggregate
  double y, z;
};
struct D3 : Base1, Base2 {  // non-homogeneous aggregate
  double y, z;
};
struct D4 : Base2, Base3 {  // homogeneous aggregate
  double y, z;
};

struct I1 : Base2 {};
struct I2 : Base2 {};
struct I3 : Base2 {};
struct D5 : I1, I2, I3 {}; // homogeneous aggregate

// PPC: define{{.*}} void @_Z7func_D12D1(%struct.D1* noalias sret(%struct.D1) align 8 %agg.result, [3 x i64] %x.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc void @_Z7func_D12D1(%struct.D1* noalias sret(%struct.D1) align 8 %agg.result, [3 x i64] %x.coerce)
// ARM64: define{{.*}} void @_Z7func_D12D1(%struct.D1* noalias sret(%struct.D1) align 8 %agg.result, %struct.D1* %x)
// X64: define dso_local x86_vectorcallcc void @"\01_Z7func_D12D1@@24"(%struct.D1* noalias sret(%struct.D1) align 8 %agg.result, %struct.D1* %x)
D1 CC func_D1(D1 x) { return x; }

// PPC: define{{.*}} [3 x double] @_Z7func_D22D2([3 x double] %x.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc %struct.D2 @_Z7func_D22D2(%struct.D2 %x.coerce)
// ARM64: define{{.*}} %struct.D2 @_Z7func_D22D2([3 x double] %x.coerce)
// X64: define dso_local x86_vectorcallcc %struct.D2 @"\01_Z7func_D22D2@@24"(%struct.D2 inreg %x.coerce)
D2 CC func_D2(D2 x) { return x; }

// PPC: define{{.*}} void @_Z7func_D32D3(%struct.D3* noalias sret(%struct.D3) align 8 %agg.result, [4 x i64] %x.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc void @_Z7func_D32D3(%struct.D3* noalias sret(%struct.D3) align 8 %agg.result, [4 x i64] %x.coerce)
// ARM64: define{{.*}} void @_Z7func_D32D3(%struct.D3* noalias sret(%struct.D3) align 8 %agg.result, %struct.D3* %x)
D3 CC func_D3(D3 x) { return x; }

// PPC: define{{.*}} [4 x double] @_Z7func_D42D4([4 x double] %x.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc %struct.D4 @_Z7func_D42D4(%struct.D4 %x.coerce)
// ARM64: define{{.*}} %struct.D4 @_Z7func_D42D4([4 x double] %x.coerce)
D4 CC func_D4(D4 x) { return x; }

D5 CC func_D5(D5 x) { return x; }
// PPC: define{{.*}} [3 x double] @_Z7func_D52D5([3 x double] %x.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc %struct.D5 @_Z7func_D52D5(%struct.D5 %x.coerce)

// The C++ multiple inheritance expansion case is a little more complicated, so
// do some extra checking.
//
// ARM64-LABEL: define{{.*}} %struct.D5 @_Z7func_D52D5([3 x double] %x.coerce)
// ARM64: bitcast %struct.D5* %{{.*}} to [3 x double]*
// ARM64: store [3 x double] %x.coerce, [3 x double]*

void call_D5(D5 *p) {
  func_D5(*p);
}

// Check the call site.
//
// ARM64-LABEL: define{{.*}} void @_Z7call_D5P2D5(%struct.D5* %p)
// ARM64: load [3 x double], [3 x double]*
// ARM64: call %struct.D5 @_Z7func_D52D5([3 x double] %{{.*}})

struct Empty { };
struct Float1 { float x; };
struct Float2 { float y; };
struct HVAWithEmptyBase : Float1, Empty, Float2 { float z; };

// PPC: define{{.*}} void @_Z15with_empty_base16HVAWithEmptyBase([3 x float] %a.coerce)
// ARM64: define{{.*}} void @_Z15with_empty_base16HVAWithEmptyBase([3 x float] %a.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc void @_Z15with_empty_base16HVAWithEmptyBase(%struct.HVAWithEmptyBase %a.coerce)
void CC with_empty_base(HVAWithEmptyBase a) {}

// FIXME: MSVC doesn't consider this an HVA because of the empty base.
// X64: define dso_local x86_vectorcallcc void @"\01_Z15with_empty_base16HVAWithEmptyBase@@16"(%struct.HVAWithEmptyBase inreg %a.coerce)

struct HVAWithEmptyBitField : Float1, Float2 {
  int : 0; // Takes no space.
  float z;
};

// PPC: define{{.*}} void @_Z19with_empty_bitfield20HVAWithEmptyBitField([3 x float] %a.coerce)
// ARM64: define{{.*}} void @_Z19with_empty_bitfield20HVAWithEmptyBitField([3 x float] %a.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc void @_Z19with_empty_bitfield20HVAWithEmptyBitField(%struct.HVAWithEmptyBitField %a.coerce)
// X64: define dso_local x86_vectorcallcc void @"\01_Z19with_empty_bitfield20HVAWithEmptyBitField@@16"(%struct.HVAWithEmptyBitField inreg %a.coerce)
void CC with_empty_bitfield(HVAWithEmptyBitField a) {}
