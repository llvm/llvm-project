// RUN: %clang_cc1 -emit-llvm -triple=x86_64-windows-msvc -fms-compatibility %s -o - | FileCheck %s

template <typename> constexpr bool _Is_integer = false;
template <> constexpr bool _Is_integer<int> = true;
template <> constexpr bool _Is_integer<char> = false;
extern "C" const bool *escape = &_Is_integer<int>;

// CHECK: @"??$_Is_integer@H@@3_NB" = linkonce_odr dso_local constant i8 1, comdat, align 1
//   Should not emit _Is_integer<char>, since it's not referenced.
// CHECK-NOT: @"??$_Is_integer@D@@3_NB"
// CHECK: @escape = dso_local global i8* @"??$_Is_integer@H@@3_NB", align 8
