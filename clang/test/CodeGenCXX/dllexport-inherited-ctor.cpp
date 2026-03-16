// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-windows-msvc -emit-llvm -std=c++20 -fms-extensions -O0 -o - %s | FileCheck --check-prefix=MSVC %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i686-windows-msvc -emit-llvm -std=c++20 -fms-extensions -O0 -o - %s | FileCheck --check-prefix=M32 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-windows-gnu -emit-llvm -std=c++20 -fms-extensions -O0 -o - %s | FileCheck --check-prefix=GNU %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-windows-msvc -emit-llvm -std=c++20 -fms-extensions -fno-dllexport-inlines -O0 -o - %s | FileCheck --check-prefix=NOINLINE %s

// Test that inherited constructors via 'using Base::Base' in a dllexport
// class are properly exported (https://github.com/llvm/llvm-project/issues/162640).

//===----------------------------------------------------------------------===//
// Basic: both base and child exported, simple parameter types.
//===----------------------------------------------------------------------===//

struct __declspec(dllexport) Base {
  Base(int);
  Base(double);
};

struct __declspec(dllexport) Child : public Base {
  using Base::Base;
};

// The inherited constructors Child(int) and Child(double) should be exported.
// Verify the thunk bodies delegate to the base constructor with the correct
// arguments, ensuring ABI compatibility with MSVC-compiled callers.

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QEAA@H@Z"(ptr {{.*}} %this, i32 %0)
// MSVC-DAG: call {{.*}} @"??0Base@@QEAA@H@Z"(ptr {{.*}} %this{{.*}}, i32
// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QEAA@N@Z"(ptr {{.*}} %this, double %0)
// MSVC-DAG: call {{.*}} @"??0Base@@QEAA@N@Z"(ptr {{.*}} %this{{.*}}, double

// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QAE@H@Z"
// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QAE@N@Z"

// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN5ChildCI14BaseEi(
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN5ChildCI14BaseEd(

//===----------------------------------------------------------------------===//
// Non-exported class should not export inherited ctors.
//===----------------------------------------------------------------------===//

struct NonExportedBase {
  NonExportedBase(int);
};

struct NonExportedChild : public NonExportedBase {
  using NonExportedBase::NonExportedBase;
};

// MSVC-NOT: dllexport{{.*}}NonExportedChild
// M32-NOT: dllexport{{.*}}NonExportedChild
// GNU-NOT: dllexport{{.*}}NonExportedChild

//===----------------------------------------------------------------------===//
// Only the derived class is dllexport, base is not.
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Multi-level inheritance: A -> B -> C with using at each level.
//===----------------------------------------------------------------------===//

struct MLBase {
  MLBase(int);
};

struct MLMiddle : MLBase {
  using MLBase::MLBase;
};

struct __declspec(dllexport) MLChild : MLMiddle {
  using MLMiddle::MLMiddle;
};

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0MLChild@@QEAA@H@Z"
// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0MLChild@@QAE@H@Z"
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN7MLChildCI16MLBaseEi(

//===----------------------------------------------------------------------===//
// Class template specialization with inherited constructors.
//===----------------------------------------------------------------------===//

template <typename T>
struct TplBase {
  TplBase(T);
};

struct __declspec(dllexport) TplChild : TplBase<int> {
  using TplBase<int>::TplBase;
};

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0TplChild@@QEAA@H@Z"
// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0TplChild@@QAE@H@Z"
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN8TplChildCI17TplBaseIiEEi(

//===----------------------------------------------------------------------===//
// Default arguments: thunk takes the full parameter list.
//===----------------------------------------------------------------------===//

struct DefArgBase {
  DefArgBase(int a, int b = 10, int c = 20);
};

struct __declspec(dllexport) DefArgChild : DefArgBase {
  using DefArgBase::DefArgBase;
};

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0DefArgChild@@QEAA@HHH@Z"(ptr {{.*}} %this, i32 %0, i32 %1, i32 %2)
// MSVC-DAG: call {{.*}} @"??0DefArgBase@@QEAA@HHH@Z"(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32

// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0DefArgChild@@QAE@HHH@Z"
// M32-DAG: call {{.*}} @"??0DefArgBase@@QAE@HHH@Z"(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32

// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN11DefArgChildCI110DefArgBaseEiii(
// GNU-DAG: call {{.*}} @_ZN10DefArgBaseC2Eiii(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32

//===----------------------------------------------------------------------===//
// Default arguments with mixed types.
//===----------------------------------------------------------------------===//

struct MixedDefBase {
  MixedDefBase(int a, double b = 3.14);
};

struct __declspec(dllexport) MixedDefChild : MixedDefBase {
  using MixedDefBase::MixedDefBase;
};

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0MixedDefChild@@QEAA@HN@Z"(ptr {{.*}} %this, i32 %0, double %1)
// MSVC-DAG: call {{.*}} @"??0MixedDefBase@@QEAA@HN@Z"(ptr {{.*}}, i32 {{.*}}, double

// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0MixedDefChild@@QAE@HN@Z"
// M32-DAG: call {{.*}} @"??0MixedDefBase@@QAE@HN@Z"(ptr {{.*}}, i32 {{.*}}, double

// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN13MixedDefChildCI112MixedDefBaseEid(
// GNU-DAG: call {{.*}} @_ZN12MixedDefBaseC2Eid(ptr {{.*}}, i32 {{.*}}, double

//===----------------------------------------------------------------------===//
// All parameters have defaults. The inherited constructor takes the full
// parameter list. The implicit default constructor also gets exported, with
// default argument values baked in.
//===----------------------------------------------------------------------===//

struct AllDefBase {
  AllDefBase(int a = 1, int b = 2);
};

struct __declspec(dllexport) AllDefChild : AllDefBase {
  using AllDefBase::AllDefBase;
};

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0AllDefChild@@QEAA@HH@Z"(ptr {{.*}} %this, i32 %0, i32 %1)
// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0AllDefChild@@QEAA@XZ"(ptr {{.*}} %this)
// MSVC-DAG: call {{.*}} @"??0AllDefBase@@QEAA@HH@Z"(ptr {{.*}}, i32 1, i32 2)

// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0AllDefChild@@QAE@HH@Z"
// M32-DAG: call {{.*}} @"??0AllDefBase@@QAE@HH@Z"(ptr {{.*}}, i32 %
// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0AllDefChild@@QAE@XZ"
// M32-DAG: call {{.*}} @"??0AllDefBase@@QAE@HH@Z"(ptr {{.*}}, i32 1, i32 2)

// GNU does not emit an implicit default constructor unless it is used.
// Only the inherited constructor with the full parameter list is exported.
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN11AllDefChildCI110AllDefBaseEii(
// GNU-DAG: call {{.*}} @_ZN10AllDefBaseC2Eii(ptr {{.*}}, i32 %{{.*}}, i32 %

//===----------------------------------------------------------------------===//
// Variadic constructor: inherited variadic constructors cannot be exported
// because delegate forwarding is not supported for variadic arguments.
//===----------------------------------------------------------------------===//

struct VariadicBase {
  VariadicBase(int, ...);
};

struct __declspec(dllexport) VariadicChild : VariadicBase {
  using VariadicBase::VariadicBase;
};

// The variadic inherited constructor (int, ...) should NOT be exported.
// Match specifically to avoid matching implicitly exported copy/move ctors.
// MSVC-NOT: dllexport{{.*}}??0VariadicChild@@QEAA@H
// M32-NOT: dllexport{{.*}}??0VariadicChild@@QAE@H
// GNU-NOT: dllexport{{.*}}VariadicChildCI1{{.*}}Eiz

//===----------------------------------------------------------------------===//
// Callee-cleanup parameter: struct with non-trivial destructor passed by value.
// canEmitDelegateCallArgs returns false for this case, so the inherited
// constructor must NOT be exported to avoid ABI incompatibility with MSVC.
//===----------------------------------------------------------------------===//

struct NontrivialDtor {
  int x;
  ~NontrivialDtor();
};

struct CalleeCleanupBase {
  CalleeCleanupBase(NontrivialDtor);
};

struct __declspec(dllexport) CalleeCleanupChild : CalleeCleanupBase {
  using CalleeCleanupBase::CalleeCleanupBase;
};

// The inherited constructor should NOT be exported on MSVC targets because the
// parameter requires callee-cleanup, making the thunk ABI-incompatible.
// On GNU targets the callee-cleanup issue does not apply, so export is fine.
// MSVC-NOT: dllexport{{.*}}CalleeCleanupChild{{.*}}NontrivialDtor
// M32-NOT: dllexport{{.*}}CalleeCleanupChild{{.*}}NontrivialDtor
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}}CalleeCleanupChild{{.*}}NontrivialDtor

//===----------------------------------------------------------------------===//
// -fno-dllexport-inlines should still export inherited constructors.
// Inherited constructors are marked inline internally but must be exported.
//===----------------------------------------------------------------------===//

// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QEAA@H@Z"
// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0Child@@QEAA@N@Z"
// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0ExportedChild@@QEAA@H@Z"
// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0ExportedChild@@QEAA@M@Z"
// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0MLChild@@QEAA@H@Z"
// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0TplChild@@QEAA@H@Z"
// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0DefArgChild@@QEAA@HHH@Z"
// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0MixedDefChild@@QEAA@HN@Z"
// NOINLINE-DAG: define weak_odr dso_local dllexport {{.*}} @"??0AllDefChild@@QEAA@HH@Z"
// The implicit default ctor is a regular inline method, NOT an inherited
// constructor, so -fno-dllexport-inlines correctly suppresses it.
// NOINLINE-NOT: define {{.*}}dllexport{{.*}} @"??0AllDefChild@@QEAA@XZ"

//===----------------------------------------------------------------------===//
// Constrained constructors: inherited constructors whose requires clause is
// not satisfied should not be exported.
// Regression test for https://github.com/llvm/llvm-project/issues/185924
//===----------------------------------------------------------------------===//

template <bool B>
struct ConstrainedBase {
  struct Enabler {};
  ConstrainedBase(Enabler) requires(B) {}
  ConstrainedBase() requires(B) : ConstrainedBase(Enabler{}) {}
  ConstrainedBase(int);
};

// B=false: both the default ctor and the Enabler ctor have requires(B) which
// is not satisfied. Only the inherited ConstrainedChild(int) should be
// exported.
struct __declspec(dllexport) ConstrainedChild : ConstrainedBase<false> {
  using ConstrainedBase::ConstrainedBase;
};

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0ConstrainedChild@@QEAA@H@Z"
// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0ConstrainedChild@@QAE@H@Z"
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN16ConstrainedChildCI115ConstrainedBaseILb0EEEi(

// The constrained constructors should NOT be exported.
// MSVC-NOT: dllexport{{.*}}ConstrainedChild@@QEAA@XZ
// M32-NOT: dllexport{{.*}}ConstrainedChild@@QAE@XZ
// GNU-NOT: dllexport{{.*}}ConstrainedBaseILb0EEEv

// Constrained non-default constructor: only export when the constraint is met.
template <typename T>
struct SelectiveBase {
  SelectiveBase(int) requires(sizeof(T) > 1) {}
  SelectiveBase(double);
};

// sizeof(char)==1, so SelectiveBase(int) requires(sizeof(char)>1) is not
// satisfied. Only the SelectiveChild(double) constructor should be exported.
struct __declspec(dllexport) SelectiveChild : SelectiveBase<char> {
  using SelectiveBase::SelectiveBase;
};

// MSVC-DAG: define weak_odr dso_local dllexport {{.*}} @"??0SelectiveChild@@QEAA@N@Z"
// M32-DAG: define weak_odr dso_local dllexport {{.*}} @"??0SelectiveChild@@QAE@N@Z"
// GNU-DAG: define {{.*}}dso_local dllexport {{.*}} @_ZN14SelectiveChildCI113SelectiveBaseIcEEd(

// The constrained int constructor should NOT be exported.
// MSVC-NOT: dllexport{{.*}}SelectiveChild@@QEAA@H@Z
// M32-NOT: dllexport{{.*}}SelectiveChild@@QAE@H@Z
// GNU-NOT: dllexport{{.*}}SelectiveBaseIcEEi

//===----------------------------------------------------------------------===//
// Non-constructor constrained method: when dllexport propagates to a base
// template specialization, methods with unsatisfied constraints should not
// be exported.
//===----------------------------------------------------------------------===//

template <typename T>
struct BaseWithConstrainedMethod {
  void foo() requires(sizeof(T) > 100) { T::nonexistent(); }
  void bar() {}
};

struct __declspec(dllexport) MethodChild : BaseWithConstrainedMethod<int> {};

// bar() should be exported (no constraint).
// MSVC-DAG: define {{.*}}dllexport {{.*}} @"?bar@?$BaseWithConstrainedMethod@H@@QEAAXXZ"
// M32-DAG: define {{.*}}dllexport {{.*}} @"?bar@?$BaseWithConstrainedMethod@H@@QAEXXZ"

// foo() should NOT be exported (constraint not satisfied).
// MSVC-NOT: dllexport{{.*}}foo@?$BaseWithConstrainedMethod@H
// M32-NOT: dllexport{{.*}}foo@?$BaseWithConstrainedMethod@H
