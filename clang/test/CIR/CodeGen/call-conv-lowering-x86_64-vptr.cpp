// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir \
// RUN:   -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir \
// RUN:   -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

struct Poly { virtual void f(); };
struct PolyLong { virtual void f(); long x; };
struct PolyTwoInt { virtual void f(); int x, y; };
struct PolyBig { virtual void f(); long x, y; };
struct PolyDerived : PolyLong { long z; };
struct HasPoly { PolyLong p; };
struct VBase { long a; };
struct VirtInherit : virtual VBase { long b; };

// A class with a virtual function has a non-trivial copy constructor, so it is
// passed by invisible reference whatever its eightbytes classify as.
int takePoly(Poly v, int k) { return k; }

// CIR: cir.func {{.*}}@_Z8takePoly4Polyi(%arg0: !cir.ptr<!rec_Poly> {llvm.align = 8 : i64, llvm.byref = !rec_Poly}{{.*}}, %arg1: !s32i {{.*}}) -> (!s32i
// LLVM-CIR: define dso_local noundef i32 @_Z8takePoly4Polyi(ptr byref(%struct.Poly) align 8 %{{[^,]+}}, i32 noundef %{{[^,)]+}})
// LLVM-OGCG: define dso_local noundef i32 @_Z8takePoly4Polyi(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(8) %{{[^,]+}}, i32 noundef %{{[^,)]+}})

int takePolyLong(PolyLong v) { return 0; }

// CIR: cir.func {{.*}}@_Z12takePolyLong8PolyLong(%arg0: !cir.ptr<!rec_PolyLong> {llvm.align = 8 : i64, llvm.byref = !rec_PolyLong}{{.*}}) -> (!s32i
// LLVM-CIR: define dso_local noundef i32 @_Z12takePolyLong8PolyLong(ptr byref(%struct.PolyLong) align 8 %{{[^,)]+}})
// LLVM-OGCG: define dso_local noundef i32 @_Z12takePolyLong8PolyLong(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(16) %{{[^,)]+}})

// The copy constructor decides this, so it does not matter what the members
// would have classified as on their own.
int takePolyTwoInt(PolyTwoInt v) { return v.y; }

// CIR: cir.func {{.*}}@_Z14takePolyTwoInt10PolyTwoInt(%arg0: !cir.ptr<!rec_PolyTwoInt> {llvm.align = 8 : i64, llvm.byref = !rec_PolyTwoInt}{{.*}}) -> (!s32i
// LLVM-CIR: define dso_local noundef i32 @_Z14takePolyTwoInt10PolyTwoInt(ptr byref(%struct.PolyTwoInt) align 8 %{{[^,)]+}})
// LLVM-OGCG: define dso_local noundef i32 @_Z14takePolyTwoInt10PolyTwoInt(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(16) %{{[^,)]+}})

// Past two eightbytes SysV says memory on its own, so the two rules agree here.
int takePolyBig(PolyBig v) { return 0; }

// CIR: cir.func {{.*}}@_Z11takePolyBig7PolyBig(%arg0: !cir.ptr<!rec_PolyBig> {llvm.align = 8 : i64, llvm.byref = !rec_PolyBig}{{.*}}) -> (!s32i
// LLVM-CIR: define dso_local noundef i32 @_Z11takePolyBig7PolyBig(ptr byref(%struct.PolyBig) align 8 %{{[^,)]+}})
// LLVM-OGCG: define dso_local noundef i32 @_Z11takePolyBig7PolyBig(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(24) %{{[^,)]+}})

// The vtable pointer is inherited through the base subobject rather than
// declared here.
int takePolyDerived(PolyDerived v) { return 0; }

// CIR: cir.func {{.*}}@_Z15takePolyDerived11PolyDerived(%arg0: !cir.ptr<!rec_PolyDerived> {llvm.align = 8 : i64, llvm.byref = !rec_PolyDerived}{{.*}}) -> (!s32i
// LLVM-CIR: define dso_local noundef i32 @_Z15takePolyDerived11PolyDerived(ptr byref(%struct.PolyDerived) align 8 %{{[^,)]+}})
// LLVM-OGCG: define dso_local noundef i32 @_Z15takePolyDerived11PolyDerived(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(24) %{{[^,)]+}})

// HasPoly declares no virtual function of its own, but its member carries the
// vtable pointer and the non-trivial copy constructor with it.
int takeHasPoly(HasPoly v) { return 0; }

// CIR: cir.func {{.*}}@_Z11takeHasPoly7HasPoly(%arg0: !cir.ptr<!rec_HasPoly> {llvm.align = 8 : i64, llvm.byref = !rec_HasPoly}{{.*}}) -> (!s32i
// LLVM-CIR: define dso_local noundef i32 @_Z11takeHasPoly7HasPoly(ptr byref(%struct.HasPoly) align 8 %{{[^,)]+}})
// LLVM-OGCG: define dso_local noundef i32 @_Z11takeHasPoly7HasPoly(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(16) %{{[^,)]+}})

// A virtual base gives the class a vtable pointer for the base offset even
// though it declares no virtual function of its own.
int takeVirtInherit(VirtInherit v) { return 0; }

// CIR: cir.func {{.*}}@_Z15takeVirtInherit11VirtInherit(%arg0: !cir.ptr<!rec_VirtInherit> {llvm.align = 8 : i64, llvm.byref = !rec_VirtInherit}{{.*}}) -> (!s32i
// LLVM-CIR: define dso_local noundef i32 @_Z15takeVirtInherit11VirtInherit(ptr byref(%struct.VirtInherit) align 8 %{{[^,)]+}})
// LLVM-OGCG: define dso_local noundef i32 @_Z15takeVirtInherit11VirtInherit(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(24) %{{[^,)]+}})

// Returning the class writes through an sret slot the caller supplies.
PolyLong makePolyLong() {
  PolyLong p;
  return p;
}

// CIR: cir.func {{.*}}@_Z12makePolyLongv(%arg0: !cir.ptr<!rec_PolyLong> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_PolyLong, llvm.writable}
// LLVM: define dso_local void @_Z12makePolyLongv(ptr dead_on_unwind noalias writable sret(%struct.PolyLong) align 8 %{{[^,)]+}})

PolyLong retPolyLong();

int caller(int k) {
  PolyLong p = retPolyLong();
  return takePolyLong(p) + k;
}

// CIR: cir.func {{.*}}@_Z6calleri(%arg0: !s32i {{.*}}) -> (!s32i
// CIR:   cir.call @_Z11retPolyLongv(%{{[0-9]+}}) : (!cir.ptr<!rec_PolyLong> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.sret = !rec_PolyLong, llvm.writable}) -> ()
// CIR:   cir.call @_Z12takePolyLong8PolyLong(%{{[0-9]+}}) : (!cir.ptr<!rec_PolyLong> {llvm.align = 8 : i64, llvm.byref = !rec_PolyLong}) -> (!s32i
// LLVM: define dso_local noundef i32 @_Z6calleri(i32 noundef %{{[^,)]+}})
// LLVM:   call void @_Z11retPolyLongv(ptr dead_on_unwind writable sret(%struct.PolyLong) align 8 %{{[^,)]+}})
// LLVM-CIR:   call noundef i32 @_Z12takePolyLong8PolyLong(ptr byref(%struct.PolyLong) align 8 %{{[^,)]+}})
// LLVM-OGCG:   call noundef i32 @_Z12takePolyLong8PolyLong(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(16) %{{[^,)]+}})

// Returned, the class uses sret at its declared alignment.
// CIR: cir.func private @_Z11retPolyLongv(!cir.ptr<!rec_PolyLong> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.sret = !rec_PolyLong, llvm.writable})
// LLVM: declare void @_Z11retPolyLongv(ptr dead_on_unwind writable sret(%struct.PolyLong) align 8)
