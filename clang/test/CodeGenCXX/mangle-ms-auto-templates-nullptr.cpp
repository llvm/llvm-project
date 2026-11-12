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

void template_mangling() {

  AutoParmTemplate<nullptr> auto_nullptr;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$M$$T0A@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$0A@@@QEAA@XZ"

  AutoFunc<nullptr>();
  // AFTER: call {{.*}} @"??$AutoFunc@$M$$T0A@@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$0A@@@YA?A?<auto>@@XZ"
}
