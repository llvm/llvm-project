// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple=x86_64-windows-msvc %s -o - | FileCheck %s

struct Foo { int x, y; };

struct S {
  // PR36125
  static constexpr char sdm_char_array[] = "asdf";

  // PR43280
  static constexpr const char *sdm_char_ptr = "asdf";

  static constexpr Foo sdm_udt{1, 2};
};

void useptr(const void *p);
void usethem() {
  useptr(&S::sdm_char_array);
  useptr(&S::sdm_char_ptr);
  useptr(&S::sdm_udt);
}

// CHECK-DAG: @"?sdm_char_array@S@@2QBDB" = linkonce_odr dso_local constant [5 x i8] c"asdf\00", comdat, align 1

// CHECK-DAG: @"?sdm_char_ptr@S@@2QEBDEB" = linkonce_odr dso_local constant i8* getelementptr inbounds ([5 x i8], [5 x i8]* @"??_C@_04JIHMPGLA@asdf?$AA@", i32 0, i32 0), comdat, align 8

// CHECK-DAG: @"?sdm_udt@S@@2UFoo@@B" = linkonce_odr dso_local constant %struct.Foo { i32 1, i32 2 }, comdat, align 4
