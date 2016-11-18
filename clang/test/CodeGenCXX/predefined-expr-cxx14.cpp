// RUN: %clang_cc1 -std=c++14 %s -triple %itanium_abi_triple -fblocks -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++14 -triple %itanium_abi_triple -fblocks -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple %itanium_abi_triple -std=c++14 -fblocks -include-pch %t %s -emit-llvm -o - | FileCheck %s

#ifndef HEADER
#define HEADER

// CHECK-DAG: @__func__._ZN13ClassTemplateIiE21classTemplateFunctionERi = private unnamed_addr constant [22 x i8] c"classTemplateFunction\00"
// CHECK-DAG: @__PRETTY_FUNCTION__._ZN13ClassTemplateIiE21classTemplateFunctionERi = private unnamed_addr constant [69 x i8] c"const auto &ClassTemplate<int>::classTemplateFunction(T &) [T = int]\00"

// CHECK-DAG: @__func__._ZN24ClassInTopLevelNamespace16functionTemplateIiEERDaRT_ = private unnamed_addr constant [17 x i8] c"functionTemplate\00"
// CHECK-DAG: @__PRETTY_FUNCTION__._ZN24ClassInTopLevelNamespace16functionTemplateIiEERDaRT_ = private unnamed_addr constant [64 x i8] c"auto &ClassInTopLevelNamespace::functionTemplate(T &) [T = int]\00"

// CHECK-DAG: @__func__._ZN24ClassInTopLevelNamespace16variadicFunctionEPiz = private unnamed_addr constant [17 x i8] c"variadicFunction\00"
// CHECK-DAG: @__PRETTY_FUNCTION__._ZN24ClassInTopLevelNamespace16variadicFunctionEPiz = private unnamed_addr constant [70 x i8] c"decltype(auto) ClassInTopLevelNamespace::variadicFunction(int *, ...)\00"

// CHECK-DAG: @__func__._ZN24ClassInTopLevelNamespace25topLevelNamespaceFunctionEv = private unnamed_addr constant [26 x i8] c"topLevelNamespaceFunction\00"
// CHECK-DAG: @__PRETTY_FUNCTION__._ZN24ClassInTopLevelNamespace25topLevelNamespaceFunctionEv = private unnamed_addr constant [60 x i8] c"auto *ClassInTopLevelNamespace::topLevelNamespaceFunction()\00"

// CHECK-DAG: @__func__.___ZN16ClassBlockConstrD2Ev_block_invoke = private unnamed_addr constant [31 x i8] c"~ClassBlockConstr_block_invoke\00"
// CHECK-DAG: @__func__.___ZN16ClassBlockConstrC2Ev_block_invoke = private unnamed_addr constant [30 x i8] c"ClassBlockConstr_block_invoke\00"

int printf(const char * _Format, ...);

class ClassInTopLevelNamespace {
public:
  auto *topLevelNamespaceFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return static_cast<int *>(nullptr);
  }

  decltype(auto) variadicFunction(int *a, ...) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return a;
  }

  template<typename T>
  auto &functionTemplate(T &t) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return t;
  }
};

template<typename T>
class ClassTemplate {
public:
  const auto &classTemplateFunction(T &t) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return t;
  }
};

struct ClassBlockConstr {
  const char *s;
  ClassBlockConstr() {
    const char * (^b)() = ^() {
    return __func__;
    };
    s = b();
  }
  ~ClassBlockConstr() {
    const char * (^b)() = ^() {
    return __func__;
    };
    s = b();
  }
};

template <class T>
class FuncTemplate {
  const char *Func;

public:
  FuncTemplate() : Func(__func__) {}
  const char *getFunc() const { return Func; }
};

int
main() {
  int a;
  ClassInTopLevelNamespace topLevelNamespace;
  ClassBlockConstr classBlockConstr;
  topLevelNamespace.topLevelNamespaceFunction();
  topLevelNamespace.variadicFunction(&a);
  topLevelNamespace.functionTemplate(a);

  ClassTemplate<int> t;
  t.classTemplateFunction(a);
  return 0;
}
#else
void Foo() {
  FuncTemplate<int> FTi;
  (void)FTi.getFunc();
}
#endif

