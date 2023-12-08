// RUN: %clang_cc1 -std=c2x -emit-llvm %s -o - | FileCheck %s

void basic_types(void) {
  auto nb = 4;          // CHECK: alloca i32
  auto dbl = 4.3;       // CHECK: alloca double
  auto lng = 4UL;       // CHECK: alloca i{{32|64}}
  auto bl = true;       // CHECK: alloca i8
  auto chr = 'A';       // CHECK: alloca i{{8|32}}
  auto str = "Test";    // CHECK: alloca ptr
  auto str2[] = "Test"; // CHECK: alloca [5 x i8]
  auto nptr = nullptr;  // CHECK: alloca ptr
}

void misc_declarations(void) {
  // FIXME: this should end up being rejected when we implement underspecified
  // declarations in N3006.
  auto strct_ptr = (struct { int a; } *)0;  // CHECK: alloca ptr
  auto int_cl = (int){13};                  // CHECK: alloca i32
  auto double_cl = (double){2.5};           // CHECK: alloca double

  auto se = ({      // CHECK: alloca i32
    auto snb = 12;  // CHECK: alloca i32
    snb;
  });
}

void loop(void) {
  auto j = 4;                       // CHECK: alloca i32
  for (auto i = j; i < 2 * j; i++); // CHECK: alloca i32
}

#define AUTO_MACRO(_NAME, ARG, ARG2, ARG3) auto _NAME = ARG + (ARG2 / ARG3);

#define AUTO_INT_MACRO(_NAME, ARG, ARG2, ARG3) auto _NAME = (ARG ^ ARG2) & ARG3;

int macros(int in_int) {
  auto a = in_int + 1;             // CHECK: alloca i32
  AUTO_MACRO(b, 1.3, 2.5f, 3);     // CHECK: alloca double
  AUTO_INT_MACRO(c, 64, 23, 0xff); // CHECK: alloca i32
  return (a + (int)b) - c;         // CHECK: ret i32 %{{.*}}
}
