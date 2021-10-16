// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -triple %itanium_abi_triple -emit-llvm -fblocks -o - %s | FileCheck %s
// RUN: %clang_cc1 -DTEST_UNALIGNED -fms-extensions -fobjc-arc -fobjc-runtime-has-weak -triple %itanium_abi_triple -emit-llvm -fblocks -o - %s | FileCheck %s --check-prefix=UNALIGNED

// CHECK-LABEL: define {{.*}}void @_Z1fPU8__strongP11objc_object(i8** noundef %0)
void f(__strong id *) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPU6__weakP11objc_object(i8** noundef %0)
void f(__weak id *) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPU15__autoreleasingP11objc_object(i8** noundef %0)
void f(__autoreleasing id *) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPP11objc_object(i8** noundef %0)
void f(__unsafe_unretained id *) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPU8__strongKP11objc_object(i8** noundef %0)
void f(const __strong id *) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPU6__weakKP11objc_object(i8** noundef %0)
void f(const __weak id *) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPU15__autoreleasingKP11objc_object(i8** noundef %0)
void f(const __autoreleasing id *) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPKP11objc_object(i8** noundef %0)
void f(const __unsafe_unretained id *) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPFU19ns_returns_retainedP11objc_objectvE
void f(__attribute__((ns_returns_retained)) id (*fn)()) {}
// CHECK-LABEL: define {{.*}}void @_Z1fP11objc_object
void f(__attribute__((ns_consumed)) id) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPFP11objc_objectU11ns_consumedS0_S0_E
void f(id (*fn)(__attribute__((ns_consumed)) id, id)) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPFP11objc_objectS0_U11ns_consumedS0_E
void f(__strong id (*fn)(id, __attribute__((ns_consumed)) id)) {}
// CHECK-LABEL: define {{.*}}void @_Z1fU13block_pointerFvU11ns_consumedP11objc_objectE
void f(void (^)(__attribute__((ns_consumed)) id)) {}

template<unsigned N> struct unsigned_c { };

// CHECK-LABEL: define weak_odr {{.*}}void @_Z1gIKvEvP10unsigned_cIXplszv1U8__bridgecvPT_v1U8__bridgecvP11objc_objectcvS3_Li0ELi1EEE
template<typename T>void g(unsigned_c<sizeof((__bridge T*)(__bridge id)(T*)0) + 1>*) {}
template void g<const void>(unsigned_c<sizeof(id) + 1> *);

#if TEST_UNALIGNED
// UNALIGNED-LABEL: define {{.*}}void @_Z1gPU6__weakU11__unalignedP11objc_object(i8** noundef %0)
void g(__weak __unaligned id *) {}
// UNALIGNED-LABEL: define {{.*}}void @_Z1gPU11__unalignedU8__strongP11objc_object(i8** noundef %0)
void g(__strong __unaligned id *) {}
// UNALIGNED-LABEL: define {{.*}}void @_Z1gPU11__unalignedU15__autoreleasingP11objc_object(i8** noundef %0)
void g(__autoreleasing __unaligned id *) {}
#endif // TEST_UNALIGNED
