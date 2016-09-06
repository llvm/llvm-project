// RUN: %clang_cc1 -std=c++1y -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2015
// RUN: %clang_cc1 -std=c++1y -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=18.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2013

template <typename> int x = 0;

// CHECK-DAG: "\01??$x@X@@3HA"
template <> int x<void>;
// CHECK-DAG: "\01??$x@H@@3HA"
template <> int x<int>;

// CHECK-DAG: "\01?FunctionWithLocalType@@YA?A?<auto>@@XZ"
auto FunctionWithLocalType() {
  struct LocalType {};
  return LocalType{};
}

// CHECK-DAG: "\01?ValueFromFunctionWithLocalType@@3ULocalType@?1??FunctionWithLocalType@@YA?A?<auto>@@XZ@A"
auto ValueFromFunctionWithLocalType = FunctionWithLocalType();

// CHECK-DAG: "\01??R<lambda_0>@@QBE?A?<auto>@@XZ"
auto LambdaWithLocalType = [] {
  struct LocalType {};
  return LocalType{};
};

// CHECK-DAG: "\01?ValueFromLambdaWithLocalType@@3ULocalType@?1???R<lambda_0>@@QBE?A?<auto>@@XZ@A"
auto ValueFromLambdaWithLocalType = LambdaWithLocalType();

template <typename T>
auto TemplateFuncionWithLocalLambda(T) {
  auto LocalLambdaWithLocalType = []() {
    struct LocalType {};
    return LocalType{};
  };
  return LocalLambdaWithLocalType();
}

// MSVC2013-DAG: "\01?ValueFromTemplateFuncionWithLocalLambda@@3ULocalType@?2???R<lambda_1>@?0???$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?3@XZ@A"
// MSVC2013-DAG: "\01?ValueFromTemplateFuncionWithLocalLambda@@3ULocalType@?2???R<lambda_1>@?0???$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?3@XZ@A"
// MSVC2015-DAG: "\01?ValueFromTemplateFuncionWithLocalLambda@@3ULocalType@?1???R<lambda_1>@?0???$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?3@XZ@A"
// MSVC2015-DAG: "\01?ValueFromTemplateFuncionWithLocalLambda@@3ULocalType@?1???R<lambda_1>@?0???$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?3@XZ@A"
// CHECK-DAG: "\01??$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z"
// CHECK-DAG: "\01??R<lambda_1>@?0???$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?1@XZ"
auto ValueFromTemplateFuncionWithLocalLambda = TemplateFuncionWithLocalLambda(0);

struct S;
template <int S::*>
int WithPMD = 0;

template <> int WithPMD<nullptr>;
// CHECK-DAG: "\01??$WithPMD@$GA@A@?0@@3HA"

template <const int *, const int *>
struct Foo {};

Foo<&x<int>, &x<int>> Zoo;
// CHECK-DAG: "\01?Zoo@@3U?$Foo@$1??$x@H@@3HA$1?1@3HA@@A"

template <typename T> T unaligned_x;
extern auto test_unaligned() { return unaligned_x<int __unaligned *>; }
// CHECK-DAG: "\01??$unaligned_x@PFAH@@3PFAHA"

