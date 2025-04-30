// RUN: %clang_cc1 -emit-llvm -O0 %s -Wno-c++2b-extensions -verify -o - 2>&1 | FileCheck %s

double sin(double x);
double cos(double x);

class FunctorClass {
public:
  __attribute__((ns_location("host"))) double operator()(int i) {
    return cos(i);
  }
};

// expected-warning@+3{{'templateFuncDeclaration': attribute can be applied only to function definitions, not declaration}}
#pragma ns mark import_recursive
template <typename T>
__attribute__((noinline)) double templateFuncDeclaration(const T &a);

// expected-warning@+1{{'funcDeclaration': attribute can be applied only to function definitions, not declaration}}
[[ns_location("grid")]] double funcDeclaration(int a);

// expected-warning@+2{{'funcDeclaration2': attribute can be applied only to function definitions, not declaration}}
#pragma ns location risc
double funcDeclaration2(int a);

int main() {
// CHECK: define internal noundef double @"_ZZ4mainENK3$_0clEi"({{.*}}) #[[ATTR_GRID:[0-9]+]]
  auto lambdaLocGrid = [] [[ns_location("grid")]] (int i)  {
    return sin(i);
  };

// CHECK: define internal noundef double @"_ZZ4mainENK3$_1clEi"({{.*}}) #[[ATTR_RISC:[0-9]+]]
  auto lambdaLocRisc = [] [[ns_location("risc")]] (int i)  {
    return sin(i);
  };

// CHECK: define internal noundef double @"_ZZ4mainENK3$_2clEi"({{.*}}) #[[ATTR_NOIMP:[0-9]+]]
  auto lambdaMarkNoImport = [] [[ns_mark("noimport")]] (int i)  {
    return sin(i);
  };

// CHECK: define internal noundef double @"_ZZ4mainENK3$_3clEi"({{.*}}) #[[ATTR_HANDOFF:[0-9]+]]
  auto lambdaMarkHandoff = [] [[ns_mark("handoff")]] (int i)  {
    return sin(i);
  };

  double nsLocRet1 = lambdaLocGrid(5);
  double nsLocRet2 = lambdaLocRisc(6);
  double nsMarkRet1 = lambdaMarkNoImport(7);
  double nsMarkRet2 = lambdaMarkHandoff(8);

// CHECK: define linkonce_odr noundef double @_ZN12FunctorClassclEi({{.*}}) #[[ATTR_HOST:[0-9]+]]
  FunctorClass funct;
  double functorRet = funct(9);

  double funcDeclarationRet = funcDeclaration(10);
  double funcDeclarationRet2 = funcDeclaration2(11);

  double funcTemplateDeclRet = templateFuncDeclaration(13);

  return 0;
}

// CHECK: attributes #[[ATTR_GRID]] = {{{.*}}"ns-location"="grid"
// CHECK: attributes #[[ATTR_RISC]] = {{{.*}}"ns-location"="risc"
// CHECK: attributes #[[ATTR_NOIMP]] = {{{.*}}"ns-mark"="noimport"
// CHECK: attributes #[[ATTR_HANDOFF]] = {{{.*}}"ns-mark"="handoff"
// CHECK: attributes #[[ATTR_HOST]] = {{{.*}}"ns-location"="host"
