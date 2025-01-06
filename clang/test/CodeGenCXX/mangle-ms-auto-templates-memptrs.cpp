// RUN: %clang_cc1 -std=c++17 -fms-compatibility-version=19.20 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=x86_64-pc-windows-msvc | FileCheck --check-prefix=AFTER %s
// RUN: %clang_cc1 -std=c++17 -fms-compatibility-version=19.14 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=x86_64-pc-windows-msvc | FileCheck --check-prefix=BEFORE %s

template <auto a>
class AutoParmTemplate {
public:
    AutoParmTemplate() {}
};

template <auto a>
auto AutoFunc() {
    return a;
}

struct A {};
struct B {};

struct S             { int a; void f(); virtual void g(); };
struct M : A, B      { int a; void f(); virtual void g(); };
struct V : virtual A { int a; void f(); virtual void g(); };

void template_mangling() {

  AutoParmTemplate<&S::f> auto_method_single_inheritance;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MP8S@@EAAXXZ1?f@1@QEAAXXZ@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$1?f@S@@QEAAXXZ@@QEAA@XZ"

  AutoParmTemplate<&M::f> auto_method_multiple_inheritance;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MP8M@@EAAXXZH?f@1@QEAAXXZA@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$H?f@M@@QEAAXXZA@@@QEAA@XZ"

  AutoParmTemplate<&V::f> auto_method_virtual_inheritance;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MP8V@@EAAXXZI?f@1@QEAAXXZA@A@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$I?f@V@@QEAAXXZA@A@@@QEAA@XZ"

  AutoFunc<&S::f>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MP8S@@EAAXXZ1?f@1@QEAAXXZ@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$1?f@S@@QEAAXXZ@@YA?A?<auto>@@XZ"

  AutoFunc<&M::f>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MP8M@@EAAXXZH?f@1@QEAAXXZA@@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$H?f@M@@QEAAXXZA@@@YA?A?<auto>@@XZ"

  AutoFunc<&V::f>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MP8V@@EAAXXZI?f@1@QEAAXXZA@A@@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$I?f@V@@QEAAXXZA@A@@@YA?A?<auto>@@XZ"

  AutoParmTemplate<&S::a> auto_data_single_inheritance;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MPEQS@@H07@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$07@@QEAA@XZ"

  AutoParmTemplate<&M::a> auto_data_multiple_inheritance;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MPEQM@@H0M@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$0M@@@QEAA@XZ"

  AutoParmTemplate<&V::a> auto_data_virtual_inheritance;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MPEQV@@HFBA@A@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$FBA@A@@@QEAA@XZ"

  AutoFunc<&S::a>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MPEQS@@H07@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$07@@YA?A?<auto>@@XZ"

  AutoFunc<&M::a>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MPEQM@@H0M@@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$0M@@@YA?A?<auto>@@XZ"

  AutoFunc<&V::a>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MPEQV@@HFBA@A@@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$FBA@A@@@YA?A?<auto>@@XZ"
}
