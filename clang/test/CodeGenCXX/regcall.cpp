// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -std=c++11     %s -o - | FileCheck -check-prefix=CHECK-LIN -check-prefix=CHECK-LIN64 %s
// RUN: %clang_cc1 -triple i386-linux-gnu -emit-llvm -std=c++11     %s -o -   | FileCheck -check-prefix=CHECK-LIN -check-prefix=CHECK-LIN32 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -std=c++11  %s -o - -DWIN_TEST | FileCheck -check-prefix=CHECK-WIN64 %s
// RUN: %clang_cc1 -triple i386-windows-msvc -emit-llvm -std=c++11  %s -o - -DWIN_TEST   | FileCheck -check-prefix=CHECK-WIN32 %s

int __regcall foo(int i);

int main()
{
  int p = 0, _data;
  auto lambda = [&](int parameter) -> int {
    _data = foo(parameter);
    return _data;
  };
  return lambda(p);
}
// CHECK-LIN: call x86_regcallcc {{.+}} @_Z15__regcall3__foo
// CHECK-WIN64: call x86_regcallcc {{.+}} @"\01?foo@@YwHH@Z"
// CHECK-WIN32: call x86_regcallcc {{.+}} @"\01?foo@@YwHH@Z"

int __regcall foo (int i){
  return i;
}
// CHECK-LIN: define x86_regcallcc {{.+}}@_Z15__regcall3__foo
// CHECK-WIN64: define dso_local x86_regcallcc {{.+}}@"\01?foo@@YwHH@Z"
// CHECK-WIN32: define dso_local x86_regcallcc {{.+}}@"\01?foo@@YwHH@Z"

// used to give a body to test_class functions
static int x = 0;
class test_class {
  int a;
public:
#ifndef WIN_TEST
  __regcall 
#endif
    test_class(){++x;}
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_classC1Ev
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_classC2Ev
  // Windows ignores calling convention on constructor/destructors.
  // CHECK-WIN64-DAG: define linkonce_odr dso_local %class.test_class* @"\01??0test_class@@QEAA@XZ"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_thiscallcc %class.test_class* @"\01??0test_class@@QAE@XZ"

#ifndef WIN_TEST
  __regcall 
#endif
  ~test_class(){--x;}
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_classD2Ev
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_classD1Ev
  // Windows ignores calling convention on constructor/destructors.
  // CHECK-WIN64-DAG: define linkonce_odr dso_local void @"\01??_Dtest_class@@QEAAXXZ"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_thiscallcc void @"\01??_Dtest_class@@QAEXXZ"
  
  test_class& __regcall operator+=(const test_class&){
    return *this;
  }
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc dereferenceable(4) %class.test_class* @_ZN10test_classpLERKS_
  // CHECK-WIN64-DAG: define linkonce_odr dso_local x86_regcallcc dereferenceable(4) %class.test_class* @"\01??Ytest_class@@QEAwAEAV0@AEBV0@@Z"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_regcallcc dereferenceable(4) %class.test_class* @"\01??Ytest_class@@QAwAAV0@ABV0@@Z"
  void __regcall do_thing(){}
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_class20__regcall3__do_thingEv
  // CHECK-WIN64-DAG: define linkonce_odr dso_local x86_regcallcc void @"\01?do_thing@test_class@@QEAwXXZ"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_regcallcc void @"\01?do_thing@test_class@@QAwXXZ"
  
  template<typename T>
  void __regcall tempFunc(T i){}
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_class20__regcall3__tempFuncIiEEvT_
  // CHECK-WIN64-DAG: define linkonce_odr dso_local x86_regcallcc void @"\01??$freeTempFunc@H@@YwXH@Z"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_regcallcc void @"\01??$freeTempFunc@H@@YwXH@Z"
};

bool __regcall operator ==(const test_class&, const test_class&){ --x; return false;}
// CHECK-LIN-DAG: define x86_regcallcc zeroext i1 @_ZeqRK10test_classS1_
// CHECK-WIN64-DAG: define dso_local x86_regcallcc zeroext i1 @"\01??8@Yw_NAEBVtest_class@@0@Z"
// CHECK-WIN32-DAG: define dso_local x86_regcallcc zeroext i1 @"\01??8@Yw_NABVtest_class@@0@Z"

test_class __regcall operator""_test_class (unsigned long long) { ++x; return test_class{};}
// CHECK-LIN64-DAG: define x86_regcallcc void @_Zli11_test_classy(%class.test_class* noalias sret %agg.result, i64)
// CHECK-LIN32-DAG: define x86_regcallcc void @_Zli11_test_classy(%class.test_class* inreg noalias sret %agg.result, i64)
// CHECK-WIN64-DAG: \01??__K_test_class@@Yw?AVtest_class@@_K@Z"
// CHECK-WIN32-DAG: \01??__K_test_class@@Yw?AVtest_class@@_K@Z"

template<typename T>
void __regcall freeTempFunc(T i){}
// CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_Z24__regcall3__freeTempFuncIiEvT_
// CHECK-WIN64-DAG: define linkonce_odr dso_local x86_regcallcc void @"\01??$freeTempFunc@H@@YwXH@Z"
// CHECK-WIN32-DAG: define linkonce_odr dso_local x86_regcallcc void @"\01??$freeTempFunc@H@@YwXH@Z"

// class to force generation of functions
void force_gen() {
  test_class t;
  test_class t2 = 12_test_class;
  t += t2;
  auto t3 = 100_test_class;
  t3.tempFunc(1);
  freeTempFunc(1);
  t3.do_thing();
}

long double _Complex __regcall foo(long double _Complex f) {
  return f;
}
// CHECK-LIN64-DAG: define x86_regcallcc void @_Z15__regcall3__fooCe({ x86_fp80, x86_fp80 }* noalias sret %agg.result, { x86_fp80, x86_fp80 }* byval align 16 %f)
// CHECK-LIN32-DAG: define x86_regcallcc void @_Z15__regcall3__fooCe({ x86_fp80, x86_fp80 }* inreg noalias sret %agg.result, { x86_fp80, x86_fp80 }* byval align 4 %f)
// CHECK-WIN64-DAG: define dso_local x86_regcallcc { double, double } @"\01?foo@@YwU?$_Complex@O@__clang@@U12@@Z"(double %f.0, double %f.1)
// CHECK-WIN32-DAG: define dso_local x86_regcallcc { double, double } @"\01?foo@@YwU?$_Complex@O@__clang@@U12@@Z"(double %f.0, double %f.1)
