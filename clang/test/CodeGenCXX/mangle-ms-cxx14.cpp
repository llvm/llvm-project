// RUN: %clang_cc1 -std=c++1y -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2015
// RUN: %clang_cc1 -std=c++1y -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=18.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2013

template <typename> int x = 0;

// CHECK: "\01??$x@X@@3HA"
template <> int x<void>;
// CHECK: "\01??$x@H@@3HA"
template <> int x<int>;

// CHECK: "\01?FunctionWithLocalType@@YA?A?<auto>@@XZ"
auto FunctionWithLocalType() {
  struct LocalType {};
  return LocalType{};
}

// CHECK: "\01?ValueFromFunctionWithLocalType@@3ULocalType@?1??FunctionWithLocalType@@YA?A?<auto>@@XZ@A"
auto ValueFromFunctionWithLocalType = FunctionWithLocalType();

// CHECK: "\01??R<lambda_0>@@QBE?A?<auto>@@XZ"
auto LambdaWithLocalType = [] {
  struct LocalType {};
  return LocalType{};
};

// CHECK: "\01?ValueFromLambdaWithLocalType@@3ULocalType@?1???R<lambda_0>@@QBE?A?<auto>@@XZ@A"
auto ValueFromLambdaWithLocalType = LambdaWithLocalType();

template <typename T>
auto TemplateFuncionWithLocalLambda(T) {
  auto LocalLambdaWithLocalType = []() {
    struct LocalType {};
    return LocalType{};
  };
  return LocalLambdaWithLocalType();
}

// MSVC2013-DAG: "\01?ValueFromTemplateFuncionWithLocalLambda@@3ULocalType@?2???R<lambda_1>@??$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?3@XZ@A"
// MSVC2013-DAG: "\01?ValueFromTemplateFuncionWithLocalLambda@@3ULocalType@?2???R<lambda_1>@??$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?3@XZ@A"
// MSVC2015-DAG: "\01?ValueFromTemplateFuncionWithLocalLambda@@3ULocalType@?1???R<lambda_1>@??$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?3@XZ@A"
// MSVC2015-DAG: "\01?ValueFromTemplateFuncionWithLocalLambda@@3ULocalType@?1???R<lambda_1>@??$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?3@XZ@A"
// CHECK: "\01??$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z"
// CHECK: "\01??R<lambda_1>@??$TemplateFuncionWithLocalLambda@H@@YA?A?<auto>@@H@Z@QBE?A?1@XZ"
auto ValueFromTemplateFuncionWithLocalLambda = TemplateFuncionWithLocalLambda(0);
