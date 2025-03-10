// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TADD -std=c++23 -fexperimental-cxx-type-aware-allocators    -fsized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TADD -std=c++23 -fexperimental-cxx-type-aware-allocators -fno-sized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TADD -std=c++23 -fexperimental-cxx-type-aware-allocators -fno-sized-deallocation -fno-aligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TADD -std=c++23 -fexperimental-cxx-type-aware-allocators    -fsized-deallocation -fno-aligned-allocation
namespace std {
  template <class T> struct type_identity {
    typedef T type;
  };
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;


template <class Tp> struct is_const {
  static const bool value = false;
};
template <class Tp> struct is_const<Tp const> {
  static const bool value = true;
};

template <class Tp> struct is_volatile {
  static const bool value = false;
};
template <class Tp> struct is_volatile<Tp volatile> {
  static const bool value = true;
};

template <class T> static const bool is_const_v = is_const<T>::value;
template <class T> static const bool is_volatile_v = is_volatile<T>::value;

struct VerifyQualifiers {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t) throw() {
    static_assert(is_const_v<T> == false); // #1
    static_assert(is_volatile_v<T> == false); // #2
    return 0;
  }
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t) {
    static_assert(is_const_v<T> == false); // #3
    static_assert(is_volatile_v<T> == false); // #4
  }
  template <typename T> void *operator new(std::type_identity<_Atomic T>, size_t, std::align_val_t) throw() {
    static_assert(is_const_v<T> == false);
    static_assert(is_volatile_v<T> == false);
  }
};

void *operator new(std::type_identity<VerifyQualifiers> type, size_t, std::align_val_t) throw() { // #11
  static_assert(is_const_v<typename decltype(type)::type> == false); // #5
  static_assert(is_volatile_v<typename decltype(type)::type> == false); // #6
  return 0;
}

void operator delete(std::type_identity<VerifyQualifiers> type, void*, size_t, std::align_val_t) {
  static_assert(is_const_v<typename decltype(type)::type> == false); // #7
  static_assert(is_volatile_v<typename decltype(type)::type> == false); // #8
}

void *operator new(std::type_identity<int>, size_t, std::align_val_t) throw() = delete; // #12
void operator delete(std::type_identity<int>, void*, size_t, std::align_val_t) = delete;

struct TestAtomic1 {

};
struct TestAtomic2 {
};

void *operator new(std::type_identity<TestAtomic1>, size_t, std::align_val_t) throw() = delete; // #13
void operator delete(std::type_identity<_Atomic TestAtomic1>, void*, size_t, std::align_val_t) = delete; // #9
void *operator new(std::type_identity<_Atomic TestAtomic2>, size_t, std::align_val_t) = delete; // #10
void operator delete(std::type_identity<TestAtomic2>, void*, size_t, std::align_val_t) = delete;

// Success tests
void test_member_allocators() {
  auto *unqualified_obj = new VerifyQualifiers();
  delete unqualified_obj;
  auto *const_obj = new const VerifyQualifiers();
  delete const_obj;
  auto *volatile_obj = new volatile VerifyQualifiers();
  delete volatile_obj;
  auto *const_volatile_obj = new const volatile VerifyQualifiers();
  delete const_volatile_obj;
  auto *atomic_obj = new _Atomic VerifyQualifiers();
  delete atomic_obj;
  auto *atomic_test1 = new _Atomic TestAtomic1;
  delete atomic_test1;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#9 {{'operator delete' has been explicitly marked deleted here}}
  auto *atomic_test2 = new _Atomic TestAtomic2;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#10 {{candidate function has been explicitly deleted}}
  // expected-note@#11 {{candidate function not viable}}
  // expected-note@#12 {{candidate function not viable}}
  // expected-note@#13 {{candidate function not viable}}
  delete atomic_test2;
}



void test_global_allocators() {
  auto *unqualified_obj = ::new VerifyQualifiers();
  ::delete unqualified_obj;
  auto *const_obj = ::new const VerifyQualifiers();
  ::delete const_obj;
  auto *volatile_obj = ::new volatile VerifyQualifiers();
  ::delete volatile_obj;
  auto *const_volatile_obj = ::new const volatile VerifyQualifiers();
  ::delete const_volatile_obj;
  _Atomic VerifyQualifiers *atomic_obj = ::new _Atomic VerifyQualifiers();
  ::delete atomic_obj;
  _Atomic int *atomic_int = new _Atomic int;
  delete atomic_int;
}
