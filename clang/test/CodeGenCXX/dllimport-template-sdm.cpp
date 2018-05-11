// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -std=c++14 -fms-extensions -o - %s | FileCheck %s --check-prefix=IMPORT
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -std=c++14 -fms-extensions -o - %s -DTEST_EXPORT | FileCheck %s --check-prefix=EXPORT

#ifndef TEST_EXPORT
#define DLLATTR __declspec(dllimport)
#else
#define DLLATTR __declspec(dllexport)
#endif

// PR37232: When a dllimport attribute is propagated from a derived class to a
// base class that happens to be a template specialization, it is only applied
// to template *methods*, and not static data members. If a dllexport attribute
// is propagated, it still applies to static data members.

// IMPORT-DAG: @"?sdm@Exporter@@2HB" = available_externally dllimport constant i32 2, align 4
// IMPORT-DAG: @"?csdm@?$A@H@@2HB" = linkonce_odr dso_local constant i32 2, comdat, align 4
// IMPORT-DAG: @"?sdm@?$A@H@@2HA" = linkonce_odr dso_local global i32 1, comdat, align 4
// IMPORT-DAG: @"?sdm@?$B@H@@2HB" = available_externally dllimport constant i32 2, align 4
// IMPORT-DAG: @"?sdm@?$C@H@@2HB" = available_externally dllimport constant i32 2, align 4

// EXPORT-DAG: @"?sdm@Exporter@@2HB" = weak_odr dso_local dllexport constant i32 2, comdat, align 4
// EXPORT-DAG: @"?csdm@?$A@H@@2HB" = weak_odr dso_local dllexport constant i32 2, comdat, align 4
// EXPORT-DAG: @"?sdm@?$A@H@@2HA" = weak_odr dso_local dllexport global i32 1, comdat, align 4
// EXPORT-DAG: @"?sdm@?$B@H@@2HB" = weak_odr dso_local dllexport constant i32 2, comdat, align 4
// EXPORT-DAG: @"?sdm@?$C@H@@2HB" = weak_odr dso_local dllexport constant i32 2, comdat, align 4


template <typename T> struct A {
  static constexpr int csdm = 2;
  static int sdm;
};
template <typename T> int A<T>::sdm = 1;

struct DLLATTR Exporter : A<int> {
  static constexpr int sdm = 2;
};

template <typename T> struct DLLATTR B { static constexpr int sdm = 2; };

template <typename T> struct DLLATTR C;
template <typename T> struct C { static constexpr int sdm = 2; };

void takeRef(const int &_Args) {}

int main() {
  takeRef(Exporter::sdm);
  takeRef(A<int>::csdm);
  takeRef(A<int>::sdm);
  takeRef(B<int>::sdm);
  takeRef(C<int>::sdm);

  return 1;
}
