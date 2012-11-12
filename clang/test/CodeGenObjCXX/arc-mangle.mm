// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -emit-llvm -o - %s | FileCheck %s

// CHECK: define void @_Z1fPU8__strongP11objc_object(i8**)
void f(__strong id *) {}
// CHECK: define void @_Z1fPU6__weakP11objc_object(i8**)
void f(__weak id *) {}
// CHECK: define void @_Z1fPU15__autoreleasingP11objc_object(i8**)
void f(__autoreleasing id *) {}
// CHECK: define void @_Z1fPP11objc_object(i8**)
void f(__unsafe_unretained id *) {}
// CHECK: define void @_Z1fPKU8__strongP11objc_object(i8**)
void f(const __strong id *) {}
// CHECK: define void @_Z1fPKU6__weakP11objc_object(i8**)
void f(const __weak id *) {}
// CHECK: define void @_Z1fPKU15__autoreleasingP11objc_object(i8**)
void f(const __autoreleasing id *) {}
// CHECK: define void @_Z1fPKP11objc_object(i8**)
void f(const __unsafe_unretained id *) {}


template<unsigned N> struct unsigned_c { };

// CHECK: define weak_odr void @_Z1gIKvEvP10unsigned_cIXplszv1U8__bridgecvPT_v1U8__bridgecvP11objc_objectcvS3_Li0ELi1EEE
template<typename T>void g(unsigned_c<sizeof((__bridge T*)(__bridge id)(T*)0) + 1>*) {}
template void g<const void>(unsigned_c<sizeof(id) + 1> *);
