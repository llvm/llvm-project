// RUN: %clang_cc1 -triple thumbv7-windows-msvc -fobjc-runtime=ios-6.0 -o - -emit-llvm %s | FileCheck %s

@protocol P;
@protocol Q;

@class I;

void f(id<P>, id, id<P>, id) {}
// CHECK-LABEL: "\01?f@@YAXPAU?$objc_object@YP@@@@PAUobjc_object@@01@Z"

void f(id, id<P>, id<P>, id) {}
// CHECK-LABEL: "\01?f@@YAXPAUobjc_object@@PAU?$objc_object@YP@@@@10@Z"

void f(id<P>, id<P>) {}
// CHECK-LABEL: "\01?f@@YAXPAU?$objc_object@YP@@@@0@Z"

void f(id<P>) {}
// CHECK-LABEL: "\01?f@@YAXPAU?$objc_object@YP@@@@@Z"

void f(id<P, Q>) {}
// CHECK-LABEL: "\01?f@@YAXPAU?$objc_object@YP@@YQ@@@@@Z"

void f(Class<P>) {}
// CHECK-LABEL: "\01?f@@YAXPAU?$objc_class@YP@@@@@Z"

void f(Class<P, Q>) {}
// CHECK-LABEL: "\01?f@@YAXPAU?$objc_class@YP@@YQ@@@@@Z"

void f(I<P> *) {}
// CHECK-LABEL: "\01?f@@YAXPAU?$I@YP@@@@@Z"

void f(I<P, Q> *) {}
// CHECK-LABEL: "\01?f@@YAXPAU?$I@YP@@YQ@@@@@Z"

