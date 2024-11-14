// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++23 -faligned-allocation -fexperimental-cxx-type-aware-allocators -fexperimental-cxx-type-aware-destroying-delete

namespace std {
  template <class T> struct type_identity {
    typedef T type;
  };
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t {};
}

using size_t = __SIZE_TYPE__;


struct __attribute__((aligned(128))) S1 {
    S1();
    template <typename T, typename U> void *operator new(std::type_identity<T>, size_t, U);
    template <typename T, typename U> void operator delete(std::type_identity<T>, void*, U); // #1
};
struct __attribute__((aligned(128))) S2 {
    S2();
    template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
    template <typename T, typename U> void operator delete(std::type_identity<T>, void*, U); // #2
};

struct __attribute__((aligned(128))) S3 {
    S3();
    template <typename T, typename U> void *operator new(std::type_identity<T>, size_t, U);
    template <typename T> void operator delete(std::type_identity<T>, void*, std::align_val_t);
};

struct __attribute__((aligned(128))) S4 {
    S4();
};

template <typename U> void *operator new(std::type_identity<S4>, size_t, U);
template <typename U> void operator delete(std::type_identity<S4>, void*, U);
// We use this deleted operator delete to verify we skip the above decl with U=align_val_t
void operator delete(std::type_identity<S4>, void*, std::align_val_t) = delete; // #3

template <typename AlignValT>
struct __attribute__((aligned(128))) S5 {
    S5();
    template <typename T> void *operator new(std::type_identity<T>, size_t, AlignValT);
    template <typename T> void operator delete(std::type_identity<T>, void*, AlignValT); // #4
};

struct __attribute__((aligned(128))) S6 {
    S6();
    template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
    template <typename T> void operator delete(std::type_identity<T>, void*, std::align_val_t); // #5
    template <typename T> void operator delete(std::type_identity<T>, S6*, std::destroying_delete_t, std::align_val_t); // #6
};

struct __attribute__((aligned(128))) S7 {
    S7();
    template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
    template <typename T> void operator delete(std::type_identity<T>, void*, std::align_val_t); // #7
    template <typename T, typename U> void operator delete(std::type_identity<T>, S7*, std::destroying_delete_t, U); // #8
    // expected-error@-1 {{destroying operator delete can have only an optional size and optional alignment parameter}}
};

template <typename AlignValT>
struct __attribute__((aligned(128))) S8 {
    S8();
    template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
    template <typename T> void operator delete(std::type_identity<T>, void*, std::align_val_t); // #9
    template <typename T> void operator delete(std::type_identity<T>, S8*, std::destroying_delete_t, AlignValT); // #10
};

extern "C" void test1() {
    S1 *s = new S1;
    delete s;
    // expected-error@-1 {{no suitable member 'operator delete' in 'S1'}}
    // expected-note@#1 {{member 'operator delete' declared here}}
}

extern "C" void test2() {
    S2 *s = new S2;
    delete s;
    // expected-error@-1 {{no suitable member 'operator delete' in 'S2'}}
    // expected-note@#2 {{member 'operator delete' declared here}}
}

extern "C" void test3() {
    S3 *s = new S3;
    delete s;
}

extern "C" void test4() {
    S4 *s = new S4;
    delete s;
    // expected-error@-1 {{attempt to use a deleted function}}
    // expected-note@#3 {{'operator delete' has been explicitly marked deleted here}}
}

extern "C" void test5() {
    S5<std::align_val_t> *s = new S5<std::align_val_t>;
    delete s;
}

extern "C" void test6() {
    S6 *s = new S6;
    delete s;
}

template <typename T> void test7_inner() {
    struct Inner {
        void *operator new(std::type_identity<Inner>, size_t);
        void operator delete(std::type_identity<Inner>, Inner*, std::destroying_delete_t, T);
    };
    Inner *obj = new Inner;
    delete obj;
}

void test7() {
    test7_inner<std::align_val_t>();
}

// extern "C" void test7() {
//     S7 *s = new S7;
//     delete s;
// }

// extern "C" void test8() {
//     S8<std::align_val_t> *s = new S8<std::align_val_t>;
//     delete s;
// }
