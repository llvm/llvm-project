// RUN: %clang_cc1 -triple thumbv7-windows-msvc -fobjc-runtime=ios-6.0 -fobjc-arc -o - -emit-llvm %s | FileCheck %s

@protocol P;
@protocol Q;

@class I;

void f(id<P>, id, id<P>, id) {}
// CHECK-LABEL: "?f@@YAXPAU?$.objc_object@U?$Protocol@UP@@@__ObjC@@@@PAU.objc_object@@01@Z"

void f(id, id<P>, id<P>, id) {}
// CHECK-LABEL: "?f@@YAXPAU.objc_object@@PAU?$.objc_object@U?$Protocol@UP@@@__ObjC@@@@10@Z"

void f(id<P>, id<P>) {}
// CHECK-LABEL: "?f@@YAXPAU?$.objc_object@U?$Protocol@UP@@@__ObjC@@@@0@Z"

void f(id<P>) {}
// CHECK-LABEL: "?f@@YAXPAU?$.objc_object@U?$Protocol@UP@@@__ObjC@@@@@Z"

void f(id<P, Q>) {}
// CHECK-LABEL: "?f@@YAXPAU?$.objc_object@U?$Protocol@UP@@@__ObjC@@U?$Protocol@UQ@@@2@@@@Z"

void f(Class<P>) {}
// CHECK-LABEL: "?f@@YAXPAU?$.objc_class@U?$Protocol@UP@@@__ObjC@@@@@Z"

void f(Class<P, Q>) {}
// CHECK-LABEL: "?f@@YAXPAU?$.objc_class@U?$Protocol@UP@@@__ObjC@@U?$Protocol@UQ@@@2@@@@Z"

void f(I<P> *) {}
// CHECK-LABEL: "?f@@YAXPAU?$.objc_cls_I@U?$Protocol@UP@@@__ObjC@@@@@Z"

void f(I<P, Q> *) {}
// CHECK-LABEL: "?f@@YAXPAU?$.objc_cls_I@U?$Protocol@UP@@@__ObjC@@U?$Protocol@UQ@@@2@@@@Z"

template <typename>
struct S {};

void f(S<__unsafe_unretained id>) {}
// CHECK-LABEL: "?f@@YAXU?$S@PAU.objc_object@@@@@Z"

void f(S<__autoreleasing id>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Autoreleasing@PAU.objc_object@@@__ObjC@@@@@Z"

void f(S<__strong id>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Strong@PAU.objc_object@@@__ObjC@@@@@Z"

void f(S<__weak id>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Weak@PAU.objc_object@@@__ObjC@@@@@Z"

void w(__weak id) {}
// CHECK-LABEL: "?w@@YAXPAU.objc_object@@@Z"

void s(__strong id) {}
// CHECK-LABEL: "?s@@YAXPAU.objc_object@@@Z"

void a(__autoreleasing id) {}
// CHECK-LABEL: "?a@@YAXPAU.objc_object@@@Z"

void u(__unsafe_unretained id) {}
// CHECK-LABEL: "?u@@YAXPAU.objc_object@@@Z"

S<__autoreleasing id> g() { return S<__autoreleasing id>(); }
// CHECK-LABEL: "?g@@YA?AU?$S@U?$Autoreleasing@PAU.objc_object@@@__ObjC@@@@XZ"

__autoreleasing id h() { return nullptr; }
// CHECK-LABEL: "?h@@YAPAU.objc_object@@XZ"
