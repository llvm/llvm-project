// RUN: %clang_cc1 %s -triple=thumbv7k-apple-watchos -emit-llvm -o - -target-abi aapcs16 | FileCheck %s
// RUN: %clang_cc1 %s -triple=thumbv7k-apple-watchos -emit-llvm -o - -target-abi aapcs16 | FileCheck -check-prefix=CHECK-GLOBALS %s

// RUN: %clang_cc1 %s -triple=arm64_32-apple-ios -emit-llvm -o - -target-abi darwinpcs | FileCheck %s
// RUN: %clang_cc1 %s -triple=arm64_32-apple-ios -emit-llvm -o - -target-abi darwinpcs | FileCheck -check-prefix=CHECK-GLOBALS %s

// __cxa_guard_acquire argument is 64-bit
struct A {
  A();
};

void f() {
  // CHECK: call i32 @__cxa_guard_acquire(ptr
  static A a;
}

// ARM64 uses the C++11 definition of POD.
namespace test1 {
  // This class is POD in C++11 and cannot have objects allocated in
  // its tail-padding.
  struct ABase {};
  struct A : ABase {
    int x;
    char c;
  };

  struct B : A {
    char d;
  };

  int test() {
    return sizeof(B);
  }
  // CHECK: define{{.*}} i32 @_ZN5test14testEv()
  // CHECK: ret i32 12
}

namespace std {
  class type_info;
}

// ARM64 uses string comparisons for what would otherwise be
// default-visibility weak RTTI.
namespace test2 {
  struct A {
    virtual void foo();
  };
  void A::foo() {}
  // Tested below because these globals get kindof oddly rearranged.

  struct __attribute__((visibility("hidden"))) B {};
  const std::type_info &b0 = typeid(B);
  // CHECK-GLOBALS: @_ZTSN5test21BE = linkonce_odr hidden constant
  // CHECK-GLOBALS: @_ZTIN5test21BE = linkonce_odr hidden constant { {{.*}}, ptr @_ZTSN5test21BE }

  const std::type_info &b1 = typeid(B*);
  // CHECK-GLOBALS: @_ZTSPN5test21BE = linkonce_odr hidden constant
  // CHECK-GLOBALS: @_ZTIPN5test21BE = linkonce_odr hidden constant { {{.*}}, ptr @_ZTSPN5test21BE, i32 0, ptr @_ZTIN5test21BE

  struct C {};
  const std::type_info &c0 = typeid(C);
  // CHECK-GLOBALS: @_ZTSN5test21CE = linkonce_odr constant [11 x i8] c"N5test21CE\00"
  // CHECK-GLOBALS: @_ZTIN5test21CE = linkonce_odr constant { {{.*}}, ptr @_ZTSN5test21CE }
}

// va_list should be based on "char *" rather than "ptr".

// CHECK: define{{.*}} void @_Z11whatsVaListPc
void whatsVaList(__builtin_va_list l) {}
