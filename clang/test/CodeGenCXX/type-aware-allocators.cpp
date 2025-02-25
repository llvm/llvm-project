// RUN: %clang_cc1 %s -triple arm64-apple-macosx    -fsized-deallocation    -faligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o -  | FileCheck --check-prefixes=CHECK,CHECK_SIZED_ALIGNED  %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx -fno-sized-deallocation    -faligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o -  | FileCheck --check-prefixes=CHECK,CHECK_NO_SIZE_ALIGNED %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx -fno-sized-deallocation -fno-aligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o -  | FileCheck --check-prefixes=CHECK,CHECK_NO_SIZE_NO_ALIGN %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx    -fsized-deallocation -fno-aligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o -  | FileCheck --check-prefixes=CHECK,CHECK_SIZED_NO_ALIGN %s


namespace std {
  template <class T> struct type_identity {
    typedef T type;
  };
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t {};
}

using size_t = __SIZE_TYPE__;

// Sanity check to esure the semantics of the selected compiler mode
// will trigger the exception handlers we are expecting, before
// involving type aware allocation.
// We duplicate the struct definitions so we don't trigger diagnostics
// for changing operator resolution on the same type, and we do the
// untyped test before specifying the typed operators rather than using
// template constraints so we don't have to deal with monstrous mangling.
struct S1 {
  S1();
};

struct __attribute__((aligned(128))) S2 {
  S2();
};

struct S3 {
  S3();
};

struct __attribute__((aligned(128))) S4 {
  S4();
  char buffer[130];
};

extern "C" void test_no_type_aware_allocator() {
  S1 *s1 = new S1;
  delete s1;
  S2 *s2 = new S2;
  delete s2;
}
// CHECK-LABEL: test_no_type_aware_allocator
// CHECK: [[ALLOC_RESULT:%.*]] = call {{.*}} @_Znwm(
// CHECK: @_ZN2S1C1Ev({{.*}} [[ALLOC_RESULT]])
// CHECK-NEXT: unwind label %[[S1LPAD:lpad]]
// CHECK_SIZED_ALIGNED: @_ZdlPvm(
// CHECK_SIZED_NO_ALIGN: @_ZdlPvm(
// CHECK_NO_SIZE_ALIGNED: @_ZdlPv(
// CHECK_NO_SIZE_NO_ALIGN: @_ZdlPv(
// CHECK_SIZED_ALIGNED: [[ALIGNED_ALLOC_RESULT:%.*]] = call {{.*}} @_ZnwmSt11align_val_t(
// CHECK_NO_SIZE_ALIGNED: [[ALIGNED_ALLOC_RESULT:%.*]] = call {{.*}} @_ZnwmSt11align_val_t(
// CHECK_SIZED_NO_ALIGN: [[ALIGNED_ALLOC_RESULT:%.*]] = call {{.*}} @_Znwm(
// CHECK_NO_SIZE_NO_ALIGN: [[ALIGNED_ALLOC_RESULT:%.*]] = call {{.*}} @_Znwm(
// CHECK: _ZN2S2C1Ev({{.*}} [[ALIGNED_ALLOC_RESULT]])
// CHECK-NEXT: unwind label %[[S2LPAD:lpad3]]
// CHECK_SIZED_ALIGNED: _ZdlPvmSt11align_val_t(
// CHECK_NO_SIZE_ALIGNED: _ZdlPvSt11align_val_t(
// CHECK_SIZED_NO_ALIGN: _ZdlPvm(
// CHECK_NO_SIZE_NO_ALIGN: _ZdlPv(
// CHECK: [[S1LPAD]]:{{.*}};
// CHECK_SIZED_ALIGNED: @_ZdlPvm({{.*}}[[ALLOC_RESULT]], {{.*}})
// CHECK_SIZED_NO_ALIGN: @_ZdlPvm({{.*}}[[ALLOC_RESULT]], {{.*}})
// CHECK_NO_SIZE_ALIGNED: @_ZdlPv({{.*}}[[ALLOC_RESULT]])
// CHECK_NO_SIZE_NO_ALIGN: @_ZdlPv({{.*}}[[ALLOC_RESULT]])
// CHECK: [[S2LPAD]]:
// CHECK_SIZED_ALIGNED: _ZdlPvSt11align_val_t({{.*}}[[ALIGNED_ALLOC_RESULT]], {{.*}})
// CHECK_NO_SIZE_ALIGNED: _ZdlPvSt11align_val_t({{.*}}[[ALIGNED_ALLOC_RESULT]], {{.*}})
// CHECK_SIZED_NO_ALIGN: _ZdlPvm({{.*}}[[ALIGNED_ALLOC_RESULT]], {{.*}})
// CHECK_NO_SIZE_NO_ALIGN: _ZdlPv({{.*}}[[ALIGNED_ALLOC_RESULT]])

template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t);

extern "C" void test_free_type_aware_allocator() {
  S3 *s3 = new S3;
  delete s3;
  S4 *s4 = new S4;
  delete s4;
}
// CHECK-LABEL: test_free_type_aware_allocator
// CHECK: [[ALLOC_RESULT:%.*]] = call {{.*}} @_ZnwI2S3EPvSt13type_identityIT_EmSt11align_val_t(
// CHECK: @_ZN2S3C1Ev({{.*}}[[ALLOC_RESULT]])
// CHECK-NEXT: unwind label %[[S3LPAD:.*]]
// CHECK: @_ZdlI2S3EvSt13type_identityIT_EPvmSt11align_val_t(
// CHECK: [[ALIGNED_ALLOC_RESULT:%.*]] = call {{.*}} @_ZnwI2S4EPvSt13type_identityIT_EmSt11align_val_t(
// CHECK: @_ZN2S4C1Ev({{.*}}[[ALIGNED_ALLOC_RESULT]])
// CHECK-NEXT: unwind label %[[S4LPAD:.*]]
// CHECK: @_ZdlI2S4EvSt13type_identityIT_EPvmSt11align_val_t({{.*}}, {{.*}}, {{.*}} 256, {{.*}} 128)
// CHECK: [[S3LPAD]]:
// CHECK: @_ZdlI2S3EvSt13type_identityIT_EPvmSt11align_val_t({{.*}}, {{.*}}[[ALLOC_RESULT]], {{.*}} 1, {{.*}} 1)
// CHECK: [[S4LPAD]]:
// CHECK: @_ZdlI2S4EvSt13type_identityIT_EPvmSt11align_val_t({{.*}}, {{.*}}[[ALIGNED_ALLOC_RESULT]], {{.*}} 256, {{.*}} 128)

struct S5 {
  S5();
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t);
  void operator delete(S5*, std::destroying_delete_t);
};

extern "C" void test_ensure_type_aware_cleanup() {
  S5 *s5 = new S5;
  delete s5;
}
// CHECK-LABEL: test_ensure_type_aware_cleanup
// CHECK: [[ALLOC_RESULT:%.*]] = call {{.*}} @_ZN2S5nwIS_EEPvSt13type_identityIT_EmSt11align_val_t(
// CHECK: @_ZN2S5C1Ev({{.*}}[[ALLOC_RESULT]])
// CHECK-NEXT: unwind label %[[S5LPAD:.*]]
// CHECK: @_ZN2S5dlEPS_St19destroying_delete_t(
// CHECK: [[S5LPAD]]:
// CHECK: @_ZN2S5dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}}, {{.*}} [[ALLOC_RESULT]]

struct S6 {
  S6();
  virtual ~S6();
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t);
};

S6::~S6(){
}
// CHECK-LABEL: _ZN2S6D0Ev
// CHECK: _ZN2S6dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t(

struct __attribute__((aligned(128))) S7 : S6 {

};


struct S8 : S6 {
  S8();
  void *operator new(size_t);
  void operator delete(void*);
};

S8::S8(){}

extern "C" void test_ensure_type_aware_overrides() {
  S6 *s6 = new S6;
  delete s6;
  S7 *s7 = new S7;
  delete s7;
  S8 *s8 = new S8;
  delete s8;
}
// CHECK-LABEL: test_ensure_type_aware_overrides
// CHECK: [[S6_ALLOC:%.*]] = call {{.*}} @_ZN2S6nwIS_EEPvSt13type_identityIT_EmSt11align_val_t(
// CHECK: @_ZN2S6C1Ev({{.*}}[[S6_ALLOC]])
// CHECK-NEXT: unwind label %[[S6LPAD:.*]]
// CHECK: [[S6_VTABLE:%vtable.*]] = load 
// CHECK: [[S6_DFN_ADDR:%.*]] = getelementptr inbounds ptr, ptr [[S6_VTABLE]], i64 1
// CHECK: [[S6_DFN:%.*]] = load ptr, ptr [[S6_DFN_ADDR]]
// CHECK: call void [[S6_DFN]](
// CHECK: [[S7_ALLOC:%.*]] = call {{.*}} @_ZN2S6nwI2S7EEPvSt13type_identityIT_EmSt11align_val_t(
// CHECK: @_ZN2S7C1Ev({{.*}}[[S7_ALLOC]])
// CHECK-NEXT: unwind label %[[S7LPAD:.*]]
// CHECK: [[S7_VTABLE:%vtable.*]] = load
// CHECK: [[S7_DFN_ADDR:%.*]] = getelementptr inbounds ptr, ptr [[S7_VTABLE]], i64 1
// CHECK: [[S7_DFN:%.*]] = load ptr, ptr [[S7_DFN_ADDR]]
// CHECK: call void [[S7_DFN]](
// CHECK: [[S8_ALLOC:%.*]] = call {{.*}} @_ZN2S8nwEm(
// CHECK: @_ZN2S8C1Ev({{.*}}[[S8_ALLOC]])
// CHECK-NEXT: unwind label %[[S8LPAD:.*]]
// CHECK: [[S8_VTABLE:%vtable.*]] = load
// CHECK: [[S8_DFN_ADDR:%.*]] = getelementptr inbounds ptr, ptr [[S8_VTABLE]], i64 1
// CHECK: [[S8_DFN:%.*]] = load ptr, ptr [[S8_DFN_ADDR]]
// CHECK: call void [[S8_DFN]](
// CHECK: [[S6LPAD]]:
// CHECK: @_ZN2S6dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}}, {{.*}} [[S6_ALLOC]], {{.*}}, {{.*}})
// CHECK: [[S7LPAD]]:
// CHECK: @_ZN2S6dlI2S7EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}}, {{.*}} [[S7_ALLOC]], {{.*}}, {{.*}})
// CHECK: [[S8LPAD]]:
// CHECK: @_ZN2S8dlEPv({{.*}} [[S8_ALLOC]])

struct __attribute__((aligned(128))) S11 {
  S11();
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t);
  void operator delete(S11*, std::destroying_delete_t, std::align_val_t);
};


struct S12 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t, unsigned line = __builtin_LINE());
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t, unsigned line = __builtin_LINE());
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t);
};

extern "C" void test_ensure_type_aware_resolution_includes_location() {
  S12 *s12 = new S12(); // test line
  delete s12;
}

// CHECK-LABEL: test_ensure_type_aware_resolution_includes_location
// `180` in the next line is the line number from the test line in above
// CHECK: %call = call noundef ptr @_ZN3S12nwIS_EEPvSt13type_identityIT_EmSt11align_val_tj({{.*}}, {{.*}}, {{.*}}, {{.*}})

// CHECK-LABEL: @_ZN2S8D0Ev
// CHECK: @_ZN2S8dlEPv(

// CHECK-LABEL: _ZN2S7D0Ev
// CHECK: _ZN2S6dlI2S7EEvSt13type_identityIT_EPvmSt11align_val_t(
