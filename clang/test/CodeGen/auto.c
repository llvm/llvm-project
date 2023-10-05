// RUN: %clang_cc1 -std=c2x -emit-llvm %s -o - | FileCheck %s

void basic_types(void) {
  auto nb = 4;          // CHECK: %nb = alloca i32, align 4
  auto dbl = 4.3;       // CHECK: %dbl = alloca double, align 8
  auto lng = 4UL;       // CHECK: %lng = alloca i{{32|64}}, align {{4|8}}
  auto bl = true;       // CHECK: %bl = alloca i8, align 1
  auto chr = 'A';       // CHECK: %chr = alloca i{{8|32}}, align {{1|4}}
  auto str = "Test";    // CHECK: %str = alloca ptr, align 8
  auto str2[] = "Test"; // CHECK: %str2 = alloca [5 x i8], align 1
  auto nptr = nullptr;  // CHECK: %nptr = alloca ptr, align 8
}

void misc_declarations(void) {
  // FIXME: this should end up being rejected when we implement underspecified
  // declarations in N3006.
  auto strct_ptr = (struct { int a; } *)0;  // CHECK: %strct_ptr = alloca ptr, align 8
  auto int_cl = (int){13};                  // CHECK: %int_cl = alloca i32, align 4
  auto double_cl = (double){2.5};           // CHECK: %double_cl = alloca double, align 8

  auto se = ({      // CHECK: %se = alloca i32, align 4
    auto snb = 12;  // CHECK: %snb = alloca i32, align 4
    snb;
  });
}

void loop(void) {
  auto j = 4;                       // CHECK: %j = alloca i32, align 4
  for (auto i = j; i < 2 * j; i++); // CHECK: %i = alloca i32, align 4
}

#define AUTO_MACRO(_NAME, ARG, ARG2, ARG3) auto _NAME = ARG + (ARG2 / ARG3);

#define AUTO_INT_MACRO(_NAME, ARG, ARG2, ARG3) auto _NAME = (ARG ^ ARG2) & ARG3;

int macros(int in_int) {
  auto a = in_int + 1;             // CHECK: %a = alloca i32, align 4
  AUTO_MACRO(b, 1.3, 2.5f, 3);     // CHECK: %b = alloca double, align 8
  AUTO_INT_MACRO(c, 64, 23, 0xff); // CHECK: %c = alloca i32, align 4
  return (a + (int)b) - c;         // CHECK: ret i32 %sub
}
