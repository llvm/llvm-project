// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-cxxabi-lowering -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR,CIR-BEFORE
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR,CIR-AFTER
struct Base1 {
  virtual ~Base1();
};

struct Base2 {
  virtual ~Base2();
};

struct Derived final : Base1 {};

using PMFTy = void(Derived::*)(void);

void untransformed_after_unreachable(Base2 &ref, PMFTy pmf) {
    auto badcast = dynamic_cast<Derived &>(ref);
    (badcast.*pmf)();

// CIR-LABEL: cir.func {{.*}}@_Z31untransformed_after_unreachableR5Base2M7DerivedFvvE
//         CIR:    %[[PMF:.*]] = cir.alloca !{{.*}} ["pmf", init]
//         CIR:    %[[DERIVED:.*]] = cir.alloca !rec_Derived
//         CIR:    cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_Base2>>, !cir.ptr<!rec_Base2>
//    CIR-NEXT:    cir.call @__cxa_bad_cast() : () -> ()
//    CIR-NEXT:    cir.unreachable
//  CIR-BEFORE:    %[[PMF_LOAD:.*]] = cir.load{{.*}} %[[PMF]]
//  CIR-BEFORE:    cir.get_method %[[PMF_LOAD]], %[[DERIVED]]
//   CIR-AFTER:    %[[PMF_LOAD:.*]] = cir.load{{.*}} %[[PMF]]
//   CIR-AFTER:    %[[PMF_EXTRACT:.*]] = cir.extract_member %[[PMF_LOAD]][1]
//   CIR-AFTER:    %[[PMF_EXTRACT:.*]] = cir.extract_member %[[PMF_LOAD]][0]
}
