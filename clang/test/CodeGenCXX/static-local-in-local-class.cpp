// RUN: %clang_cc1 -triple x86_64-linux -fblocks -emit-llvm -o - %s -std=c++1y | FileCheck %s

// CHECK: @"_ZZZNK17pr18020_constexpr3$_1clEvENKUlvE_clEvE2l2" =
// CHECK: internal global i32* @"_ZZNK17pr18020_constexpr3$_1clEvE2l1"
// CHECK: @_ZZL14deduced_returnvE1n = internal global i32 42
// CHECK: @_ZZZL20block_deduced_returnvEUb_E1n = internal global i32 42
// CHECK: @_ZZ18static_local_labelPvE1q = linkonce_odr global i8* blockaddress(@_Z18static_local_labelPv, %{{.*}})
// CHECK: @"_ZZNK3$_2clEvE1x" = internal global i32 42

namespace pr6769 {
struct X {
  static void f();
};

void X::f() {
  static int *i;
  {
    struct Y {
      static void g() {
        i = new int();
	*i = 100;
	(*i) = (*i) +1;
      }
    };
    (void)Y::g();
  }
  (void)i;
}
}

namespace pr7101 {
void foo() {
    static int n = 0;
    struct Helper {
        static void Execute() {
            n++;
        }
    };
    Helper::Execute();
}
}

// These tests all break the assumption that the static var decl has to be
// emitted before use of the var decl.  This happens because we defer emission
// of variables with internal linkage and no initialization side effects, such
// as 'x'.  Then we hit operator()() in 'f', and emit the callee before we emit
// the arguments, so we emit the innermost function first.

namespace pr18020_lambda {
// Referring to l1 before emitting it used to crash.
auto x = []() {
  static int l1 = 0;
  return [] { return l1; };
};
int f() { return x()(); }
}

// CHECK-LABEL: define internal noundef i32 @"_ZZNK14pr18020_lambda3$_0clEvENKUlvE_clEv"
// CHECK: load i32, i32* @"_ZZNK14pr18020_lambda3$_0clEvE2l1"

namespace pr18020_constexpr {
// Taking the address of l1 in a constant expression used to crash.
auto x = []() {
  static int l1 = 0;
  return [] {
    static int *l2 = &l1;
    return *l2;
  };
};
int f() { return x()(); }
}

// CHECK-LABEL: define internal noundef i32 @"_ZZNK17pr18020_constexpr3$_1clEvENKUlvE_clEv"
// CHECK: load i32*, i32** @"_ZZZNK17pr18020_constexpr3$_1clEvENKUlvE_clEvE2l2"

// Lambda-less reduction that references l1 before emitting it.  This didn't
// crash if you put it in a namespace.
struct pr18020_class {
  auto operator()() {
    static int l1 = 0;
    struct U {
      int operator()() { return l1; }
    };
    return U();
  }
};
static pr18020_class x;
int pr18020_f() { return x()(); }

// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZN13pr18020_classclEvEN1UclEv
// CHECK: load i32, i32* @_ZZN13pr18020_classclEvE2l1

// In this test case, the function containing the static local will not be
// emitted because it is unneeded. However, the operator call of the inner class
// is called, and the static local is referenced and must be emitted.
static auto deduced_return() {
  static int n = 42;
  struct S { int *operator()() { return &n; } };
  return S();
}
extern "C" int call_deduced_return_operator() {
  return *decltype(deduced_return())()();
}

// CHECK-LABEL: define{{.*}} i32 @call_deduced_return_operator()
// CHECK: call noundef i32* @_ZZL14deduced_returnvEN1SclEv(
// CHECK: load i32, i32* %
// CHECK: ret i32 %

// CHECK-LABEL: define internal noundef i32* @_ZZL14deduced_returnvEN1SclEv(%struct.S* {{[^,]*}} %this)
// CHECK: ret i32* @_ZZL14deduced_returnvE1n

static auto block_deduced_return() {
  auto (^b)() = ^() {
    static int n = 42;
    struct S { int *operator()() { return &n; } };
    return S();
  };
  return b();
}
extern "C" int call_block_deduced_return() {
  return *decltype(block_deduced_return())()();
}

// CHECK-LABEL: define{{.*}} i32 @call_block_deduced_return()
// CHECK: call noundef i32* @_ZZZL20block_deduced_returnvEUb_EN1SclEv(
// CHECK: load i32, i32* %
// CHECK: ret i32 %

// CHECK-LABEL: define internal noundef i32* @_ZZZL20block_deduced_returnvEUb_EN1SclEv(%struct.S.6* {{[^,]*}} %this) #1 align 2 {
// CHECK: ret i32* @_ZZZL20block_deduced_returnvEUb_E1n

inline auto static_local_label(void *p) {
  if (p)
    goto *p;
  static void *q = &&label;
  struct S { static void *get() { return q; } };
  return S();
label:
  __builtin_abort();
}
void *global_label = decltype(static_local_label(0))::get();

// CHECK-LABEL: define linkonce_odr noundef i8* @_ZZ18static_local_labelPvEN1S3getEv()
// CHECK: %[[lbl:[^ ]*]] = load i8*, i8** @_ZZ18static_local_labelPvE1q
// CHECK: ret i8* %[[lbl]]

auto global_lambda = []() {
  static int x = 42;
  struct S { static int *get() { return &x; } };
  return S();
};
extern "C" int use_global_lambda() {
  return *decltype(global_lambda())::get();
}
// CHECK-LABEL: define{{.*}} i32 @use_global_lambda()
// CHECK: call noundef i32* @"_ZZNK3$_2clEvEN1S3getEv"()

// CHECK-LABEL: define internal noundef i32* @"_ZZNK3$_2clEvEN1S3getEv"()
// CHECK: ret i32* @"_ZZNK3$_2clEvE1x"
