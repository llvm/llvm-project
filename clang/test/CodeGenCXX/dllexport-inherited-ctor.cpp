// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-windows-msvc -emit-llvm -std=c++17 -fms-extensions -O0 -o - %s | FileCheck --check-prefix=MSVC %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i686-windows-msvc -emit-llvm -std=c++17 -fms-extensions -O0 -o - %s | FileCheck --check-prefix=M32 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-windows-gnu -emit-llvm -std=c++17 -fms-extensions -O0 -o - %s | FileCheck --check-prefix=GNU %s

// Test that inherited constructors via 'using Base::Base' in a dllexport
// class are properly exported (https://github.com/llvm/llvm-project/issues/162640).

struct __declspec(dllexport) Base {
  Base(int);
  Base(double);
};

struct __declspec(dllexport) Child : public Base {
  using Base::Base;
};

// The inherited constructors Child(int) and Child(double) should be exported.

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QEAA@H@Z"
// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QEAA@N@Z"

// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QAE@H@Z"
// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QAE@N@Z"

// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN5ChildCI14BaseEi(
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN5ChildCI14BaseEd(

// Also test that a non-exported derived class does not export inherited ctors.
struct NonExportedBase {
  NonExportedBase(int);
};

struct NonExportedChild : public NonExportedBase {
  using NonExportedBase::NonExportedBase;
};

// MSVC-NOT: dllexport{{.*}}NonExportedChild
// M32-NOT: dllexport{{.*}}NonExportedChild
// GNU-NOT: dllexport{{.*}}NonExportedChild

// Test that only the derived class is dllexport, base is not.
struct PlainBase {
  PlainBase(int);
  PlainBase(float);
};

struct __declspec(dllexport) ExportedChild : public PlainBase {
  using PlainBase::PlainBase;
};

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0ExportedChild@@QEAA@H@Z"
// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0ExportedChild@@QEAA@M@Z"

// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0ExportedChild@@QAE@H@Z"
// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0ExportedChild@@QAE@M@Z"

// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN13ExportedChildCI19PlainBaseEi(
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN13ExportedChildCI19PlainBaseEf(
