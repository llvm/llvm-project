// RUN: %clang_cc1 %s -o - -emit-llvm -std=c++23 -faligned-allocation -fexperimental-cxx-type-aware-allocators -fexceptions | FileCheck --check-prefixes=CHECK,CHECK_TA %s
// RUN: %clang_cc1 %s -o - -emit-llvm -std=c++23 -faligned-allocation                                          -fexceptions | FileCheck %s

// This is a semantic test, but the only way to observe the pickup of the usual delete
// for a constructor call's clean up

namespace std {
  template <class T> struct type_identity {
    typedef T type;
  };
  enum class align_val_t : __SIZE_TYPE__ {};
}

using size_t = __SIZE_TYPE__;


struct __attribute__((aligned(128))) S1 {
    S1();
#if __has_feature(cxx_type_aware_allocators)
    template <typename T, typename U> void *operator new(std::type_identity<T>, size_t, U) asm("S1_operator_new");
    template <typename T, typename U> void operator delete(std::type_identity<T>, void*, U) asm("S1_cleanup_operator_delete");
#else
    template <typename U> void *operator new(size_t, U) asm("S1_operator_new");
    template <typename U> void operator delete(void*, U) asm("S1_cleanup_operator_delete");
#endif
};
struct __attribute__((aligned(128))) S2 {
    S2();
#if __has_feature(cxx_type_aware_allocators)
    template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t) asm("S2_operator_new");
    template <typename T, typename U> void operator delete(std::type_identity<T>, void*, U) asm("S2_cleanup_operator_delete");
#else
    void *operator new(size_t, std::align_val_t) asm("S2_operator_new");
    template <typename U> void operator delete(void*, U) asm("S2_cleanup_operator_delete");
#endif
};

struct __attribute__((aligned(128))) S3 {
    S3();
#if __has_feature(cxx_type_aware_allocators)
    template <typename T, typename U> void *operator new(std::type_identity<T>, size_t, U) asm("S3_operator_new");
    template <typename T> void operator delete(std::type_identity<T>, void*, std::align_val_t) asm("S3_cleanup_operator_delete");
#else
    template <typename U> void *operator new(size_t, U) asm("S3_operator_new");
    void operator delete(void*, std::align_val_t) asm("S3_cleanup_operator_delete");
#endif
};

#if __has_feature(cxx_type_aware_allocators)
struct __attribute__((aligned(128))) S4 {
    S4();
};

template <typename U> void *operator new(std::type_identity<S4>, size_t, U) asm("S4_operator_new");
template <typename U> void operator delete(std::type_identity<S4>, void*, U) asm("S4_cleanup_operator_delete");
#endif

template <typename AlignValT>
struct __attribute__((aligned(128))) S5 {
    S5();
#if __has_feature(cxx_type_aware_allocators)
    template <typename T> void *operator new(std::type_identity<T>, size_t, AlignValT) asm("S5_operator_new");
    template <typename T> void operator delete(std::type_identity<T>, void*, AlignValT) asm("S5_cleanup_operator_delete");
#else
    void *operator new(size_t, AlignValT) asm("S5_operator_new");
    void operator delete(void*, AlignValT) asm("S5_cleanup_operator_delete");
#endif
};


extern "C" void test1() {
    S1 *s = new S1;
// CHECK-LABEL: test1
// CHECK: S1_operator_new
// CHECK: S1_cleanup_operator_delete
}

extern "C" void test2() {
    S2 *s = new S2;
// CHECK-LABEL: test2
// CHECK: S2_operator_new
// CHECK: S2_cleanup_operator_delete
}

extern "C" void test3() {
    S3 *s = new S3;
// CHECK-LABEL: test3
// CHECK: S3_operator_new
// CHECK: S3_cleanup_operator_delete
}

#if __has_feature(cxx_type_aware_allocators)
extern "C" void test4() {
    S4 *s = new S4;
// CHECK_TA-LABEL: test4
// CHECK_TA: S4_operator_new
// CHECK_TA: S4_cleanup_operator_delete
}
#endif

extern "C" void test5() {
    S5<std::align_val_t> *s = new S5<std::align_val_t>;
// CHECK-LABEL: test5
// CHECK: S5_operator_new
// CHECK: S5_cleanup_operator_delete
}
