// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC
// RUN: %clang_cc1 -no-opaque-pointers -mfloat-abi hard -triple armv7-unknown-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM32
// RUN: %clang_cc1 -no-opaque-pointers -mfloat-abi hard -triple aarch64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM64
// RUN: %clang_cc1 -no-opaque-pointers -mfloat-abi hard -triple x86_64-unknown-windows-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 -no-opaque-pointers -mfloat-abi hard -triple aarch64-unknown-windows-msvc -emit-llvm -o - %s | FileCheck %s --check-prefix=WOA64

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
// ARM64: define{{.*}} void @_Z7func_D12D1(%struct.D1* noalias sret(%struct.D1) align 8 %agg.result, %struct.D1* noundef %x)
// X64: define dso_local x86_vectorcallcc void @"\01_Z7func_D12D1@@24"(%struct.D1* noalias sret(%struct.D1) align 8 %agg.result, %struct.D1* noundef %x)
D1 CC func_D1(D1 x) { return x; }

// PPC: define{{.*}} [3 x double] @_Z7func_D22D2([3 x double] %x.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc %struct.D2 @_Z7func_D22D2(%struct.D2 %x.coerce)
// ARM64: define{{.*}} %struct.D2 @_Z7func_D22D2([3 x double] %x.coerce)
// X64: define dso_local x86_vectorcallcc %struct.D2 @"\01_Z7func_D22D2@@24"(%struct.D2 inreg %x.coerce)
D2 CC func_D2(D2 x) { return x; }

// PPC: define{{.*}} void @_Z7func_D32D3(%struct.D3* noalias sret(%struct.D3) align 8 %agg.result, [4 x i64] %x.coerce)
// ARM32: define{{.*}} arm_aapcs_vfpcc void @_Z7func_D32D3(%struct.D3* noalias sret(%struct.D3) align 8 %agg.result, [4 x i64] %x.coerce)
// ARM64: define{{.*}} void @_Z7func_D32D3(%struct.D3* noalias sret(%struct.D3) align 8 %agg.result, %struct.D3* noundef %x)
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
// ARM64-LABEL: define{{.*}} void @_Z7call_D5P2D5(%struct.D5* noundef %p)
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

// WOA64: define dso_local void @"?with_empty_base@@YAXUHVAWithEmptyBase@@@Z"([2 x i64] %{{.*}})
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

namespace pr47611 {
// MSVC on Arm includes "isCXX14Aggregate" as part of its definition of
// Homogeneous Floating-point Aggregate (HFA). Additionally, it has a different
// handling of C++14 aggregates, which can lead to confusion.

// Pod is a trivial HFA.
struct Pod {
  double b[2];
};
// Not an aggregate according to C++14 spec => not HFA according to MSVC.
struct NotCXX14Aggregate {
  NotCXX14Aggregate();
  Pod p;
};
// NotPod is a C++14 aggregate. But not HFA, because it contains
// NotCXX14Aggregate (which itself is not HFA because it's not a C++14
// aggregate).
struct NotPod {
  NotCXX14Aggregate x;
};
struct Empty {};
// A class with a base is returned using the sret calling convetion by MSVC.
struct HasEmptyBase : public Empty {
  double b[2];
};
struct HasPodBase : public Pod {};
// WOA64-LABEL: define dso_local %"struct.pr47611::Pod" @"?copy@pr47611@@YA?AUPod@1@PEAU21@@Z"(%"struct.pr47611::Pod"* noundef %x)
Pod copy(Pod *x) { return *x; } // MSVC: ldp d0,d1,[x0], Clang: ldp d0,d1,[x0]
// WOA64-LABEL: define dso_local void @"?copy@pr47611@@YA?AUNotCXX14Aggregate@1@PEAU21@@Z"(%"struct.pr47611::NotCXX14Aggregate"* inreg noalias sret(%"struct.pr47611::NotCXX14Aggregate") align 8 %agg.result, %"struct.pr47611::NotCXX14Aggregate"* noundef %x)
NotCXX14Aggregate copy(NotCXX14Aggregate *x) { return *x; } // MSVC: stp x8,x9,[x0], Clang: str q0,[x0]
// WOA64-LABEL: define dso_local [2 x i64] @"?copy@pr47611@@YA?AUNotPod@1@PEAU21@@Z"(%"struct.pr47611::NotPod"* noundef %x)
NotPod copy(NotPod *x) { return *x; }
// WOA64-LABEL: define dso_local void @"?copy@pr47611@@YA?AUHasEmptyBase@1@PEAU21@@Z"(%"struct.pr47611::HasEmptyBase"* inreg noalias sret(%"struct.pr47611::HasEmptyBase") align 8 %agg.result, %"struct.pr47611::HasEmptyBase"* noundef %x)
HasEmptyBase copy(HasEmptyBase *x) { return *x; }
// WOA64-LABEL: define dso_local void @"?copy@pr47611@@YA?AUHasPodBase@1@PEAU21@@Z"(%"struct.pr47611::HasPodBase"* inreg noalias sret(%"struct.pr47611::HasPodBase") align 8 %agg.result, %"struct.pr47611::HasPodBase"* noundef %x)
HasPodBase copy(HasPodBase *x) { return *x; }

void call_copy_pod(Pod *pod) {
  *pod = copy(pod);
  // WOA64-LABEL: define dso_local void @"?call_copy_pod@pr47611@@YAXPEAUPod@1@@Z"
  // WOA64: %{{.*}} = call %"struct.pr47611::Pod" @"?copy@pr47611@@YA?AUPod@1@PEAU21@@Z"(%"struct.pr47611::Pod"* noundef %{{.*}})
}

void call_copy_notcxx14aggregate(NotCXX14Aggregate *notcxx14aggregate) {
  *notcxx14aggregate = copy(notcxx14aggregate);
  // WOA64-LABEL: define dso_local void @"?call_copy_notcxx14aggregate@pr47611@@YAXPEAUNotCXX14Aggregate@1@@Z"
  // WOA64: call void @"?copy@pr47611@@YA?AUNotCXX14Aggregate@1@PEAU21@@Z"(%"struct.pr47611::NotCXX14Aggregate"* inreg sret(%"struct.pr47611::NotCXX14Aggregate") align 8 %{{.*}}, %"struct.pr47611::NotCXX14Aggregate"* noundef %{{.*}})
}

void call_copy_notpod(NotPod *notPod) {
  *notPod = copy(notPod);
  // WOA64-LABEL: define dso_local void @"?call_copy_notpod@pr47611@@YAXPEAUNotPod@1@@Z"
  // WOA64: %{{.*}} = call [2 x i64] @"?copy@pr47611@@YA?AUNotPod@1@PEAU21@@Z"(%"struct.pr47611::NotPod"* noundef %{{.*}})
}

void call_copy_hasemptybase(HasEmptyBase *hasEmptyBase) {
  *hasEmptyBase = copy(hasEmptyBase);
  // WOA64-LABEL: define dso_local void @"?call_copy_hasemptybase@pr47611@@YAXPEAUHasEmptyBase@1@@Z"
  // WOA64: call void @"?copy@pr47611@@YA?AUHasEmptyBase@1@PEAU21@@Z"(%"struct.pr47611::HasEmptyBase"* inreg sret(%"struct.pr47611::HasEmptyBase") align 8 %{{.*}}, %"struct.pr47611::HasEmptyBase"* noundef %{{.*}})
}

void call_copy_haspodbase(HasPodBase *hasPodBase) {
  *hasPodBase = copy(hasPodBase);
  // WOA64-LABEL: define dso_local void @"?call_copy_haspodbase@pr47611@@YAXPEAUHasPodBase@1@@Z"
  // WOA64: call void @"?copy@pr47611@@YA?AUHasPodBase@1@PEAU21@@Z"(%"struct.pr47611::HasPodBase"* inreg sret(%"struct.pr47611::HasPodBase") align 8 %{{.*}}, %"struct.pr47611::HasPodBase"* noundef %{{.*}})
}
} // namespace pr47611

namespace protected_member {
struct HFA {
  double x;
  double y;
protected:
  double z;
};
double foo(HFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@protected_member@@YANUHFA@1@@Z"([3 x double] %{{.*}})
}
namespace private_member {
struct HFA {
  double x;
  double y;
private:
  double z;
};
double foo(HFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@private_member@@YANUHFA@1@@Z"([3 x double] %{{.*}})
}
namespace polymorphic {
struct NonHFA {
  double x;
  double y;
  double z;
  virtual void f1();
};
double foo(NonHFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@polymorphic@@YANUNonHFA@1@@Z"(%"struct.polymorphic::NonHFA"* noundef %{{.*}})
}
namespace trivial_copy_assignment {
struct HFA {
  double x;
  double y;
  double z;
  HFA &operator=(const HFA&) = default;
};
double foo(HFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@trivial_copy_assignment@@YANUHFA@1@@Z"([3 x double] %{{.*}})
}
namespace non_trivial_copy_assignment {
struct NonHFA {
  double x;
  double y;
  double z;
  NonHFA &operator=(const NonHFA&);
};
double foo(NonHFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@non_trivial_copy_assignment@@YANUNonHFA@1@@Z"(%"struct.non_trivial_copy_assignment::NonHFA"* noundef %{{.*}})
}
namespace user_provided_ctor {
struct HFA {
  double x;
  double y;
  double z;
  HFA(int);
};
double foo(HFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@user_provided_ctor@@YANUHFA@1@@Z"([3 x double] %{{.*}})
}
namespace trivial_dtor {
struct HFA {
  double x;
  double y;
  double z;
  ~HFA() = default;
};
double foo(HFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@trivial_dtor@@YANUHFA@1@@Z"([3 x double] %{{.*}})
}
namespace non_trivial_dtor {
struct NonHFA {
  double x;
  double y;
  double z;
  ~NonHFA();
};
double foo(NonHFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@non_trivial_dtor@@YANUNonHFA@1@@Z"(%"struct.non_trivial_dtor::NonHFA"* noundef %{{.*}})
}
namespace non_empty_base {
struct non_empty_base { double d; };
struct HFA : non_empty_base {
  double x;
  double y;
  double z;
};
double foo(HFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@non_empty_base@@YANUHFA@1@@Z"([4 x double] %{{.*}})
}
namespace empty_field {
struct empty { };
struct NonHFA {
  double x;
  double y;
  double z;
  empty e;
};
double foo(NonHFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@empty_field@@YANUNonHFA@1@@Z"(%"struct.empty_field::NonHFA"* noundef %{{.*}})
}
namespace non_empty_field {
struct non_empty { double d; };
struct HFA {
  double x;
  double y;
  double z;
  non_empty e;
};
double foo(HFA v) { return v.x + v.y; }
// WOA64: define dso_local noundef double @"?foo@non_empty_field@@YANUHFA@1@@Z"([4 x double] %{{.*}})
}
