// RUN: %clang_cc1 -std=c++17 -fms-compatibility-version=19.20 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=x86_64-pc-windows-msvc | FileCheck %s

struct StructA {};

template<class T>
auto AutoT() { return T(); }

template<class T>
const auto AutoConstT() { return T(); }

template<class T>
volatile auto AutoVolatileT() { return T(); }

template<class T>
const volatile auto AutoConstVolatileT() { return T(); }

// The qualifiers of the return type should always be emitted even for void types.
// Void types usually have their qualifers stripped in the mangled name for MSVC ABI.
void test_template_auto_void() {
  AutoT<void>();
  // CHECK: call {{.*}} @"??$AutoT@X@@YA?A_PXZ"

  AutoT<const void>();
  // CHECK: call {{.*}} @"??$AutoT@$$CBX@@YA?A_PXZ"

  AutoT<volatile void>();
  // CHECK: call {{.*}} @"??$AutoT@$$CCX@@YA?A_PXZ"

  AutoT<const volatile void>();
  // CHECK: call {{.*}} @"??$AutoT@$$CDX@@YA?A_PXZ"

  AutoConstT<void>();
  // CHECK: call {{.*}} @"??$AutoConstT@X@@YA?B_PXZ"

  AutoVolatileT<void>();
  // CHECK: call {{.*}} @"??$AutoVolatileT@X@@YA?C_PXZ"

  AutoConstVolatileT<void>();
  // CHECK: call {{.*}} @"??$AutoConstVolatileT@X@@YA?D_PXZ"
}

void test_template_auto_int() {
  AutoT<int>();
  // CHECK: call {{.*}} @"??$AutoT@H@@YA?A_PXZ"

  AutoT<const int>();
  // CHECK: call {{.*}} @"??$AutoT@$$CBH@@YA?A_PXZ"

  AutoT<volatile int>();
  // CHECK: call {{.*}} @"??$AutoT@$$CCH@@YA?A_PXZ"

  AutoT<const volatile int>();
  // CHECK: call {{.*}} @"??$AutoT@$$CDH@@YA?A_PXZ"

  AutoConstT<int>();
  // CHECK: call {{.*}} @"??$AutoConstT@H@@YA?B_PXZ"

  AutoVolatileT<int>();
  // CHECK: call {{.*}} @"??$AutoVolatileT@H@@YA?C_PXZ"

  AutoConstVolatileT<int>();
  // CHECK: call {{.*}} @"??$AutoConstVolatileT@H@@YA?D_PXZ"
}

void test_template_auto_struct() {
  AutoT<StructA>();
  // CHECK: call {{.*}} @"??$AutoT@UStructA@@@@YA?A_PXZ"

  AutoT<const StructA>();
  // CHECK: call {{.*}} @"??$AutoT@$$CBUStructA@@@@YA?A_PXZ"

  AutoConstT<StructA>();
  // CHECK: call {{.*}} @"??$AutoConstT@UStructA@@@@YA?B_PXZ"

  AutoVolatileT<StructA>();
  // CHECK: call {{.*}} @"??$AutoVolatileT@UStructA@@@@YA?C_PXZ"

  AutoConstVolatileT<StructA>();
  // CHECK: call {{.*}} @"??$AutoConstVolatileT@UStructA@@@@YA?D_PXZ"
}

void test_template_auto_ptr() {
  AutoT<int*>();
  // CHECK: call {{.*}} @"??$AutoT@PEAH@@YA?A_PXZ"

  AutoT<const int*>();
  // CHECK: call {{.*}} @"??$AutoT@PEBH@@YA?A_PXZ"

  AutoT<const int* const>();
  // CHECK: call {{.*}} @"??$AutoT@QEBH@@YA?A_PXZ"

  AutoConstT<int*>();
  // CHECK: call {{.*}} @"??$AutoConstT@PEAH@@YA?B_PXZ"

  AutoVolatileT<int*>();
  // CHECK: call {{.*}} @"??$AutoVolatileT@PEAH@@YA?C_PXZ"

  AutoConstVolatileT<int*>();
  // CHECK: call {{.*}} @"??$AutoConstVolatileT@PEAH@@YA?D_PXZ"
}

template<class T>
auto* PtrAutoT() { return T(); }

template<class T>
const auto* PtrAutoConstT() { return T(); }

template<class T>
volatile auto* PtrAutoVolatileT() { return T(); }

template<class T>
const volatile auto* PtrAutoConstVolatileT() { return T(); }

void test_template_ptr_auto() {
  PtrAutoT<int*>();
  // CHECK: call {{.*}} @"??$PtrAutoT@PEAH@@YAPEA_PXZ"

  PtrAutoT<const int*>();
  // CHECK: call {{.*}} @"??$PtrAutoT@PEBH@@YAPEA_PXZ"

  PtrAutoT<const int* const>();
  // CHECK: call {{.*}} @"??$PtrAutoT@QEBH@@YAPEA_PXZ"

  PtrAutoConstT<int*>();
  // CHECK: call {{.*}} @"??$PtrAutoConstT@PEAH@@YAPEB_PXZ"

  PtrAutoVolatileT<int*>();
  // CHECK: call {{.*}} @"??$PtrAutoVolatileT@PEAH@@YAPEC_PXZ"

  PtrAutoConstVolatileT<int*>();
  // CHECK: call {{.*}} @"??$PtrAutoConstVolatileT@PEAH@@YAPED_PXZ"
}

int func_int();
const int func_constint();
void func_void();
int* func_intptr();

template<class T, T v>
auto (*FuncPtrAutoT())() { return v; }

void test_template_func_ptr_auto() {
  FuncPtrAutoT<int (*)(), &func_int>();
  // CHECK: call {{.*}} @"??$FuncPtrAutoT@P6AHXZ$1?func_int@@YAHXZ@@YAP6A?A_PXZXZ"

  FuncPtrAutoT<const int (*)(), &func_constint>();
  // CHECK: call {{.*}} @"??$FuncPtrAutoT@P6A?BHXZ$1?func_constint@@YA?BHXZ@@YAP6A?A_PXZXZ"

  FuncPtrAutoT<void (*)(), &func_void>();
  // CHECK: call {{.*}} @"??$FuncPtrAutoT@P6AXXZ$1?func_void@@YAXXZ@@YAP6A?A_PXZXZ"

  FuncPtrAutoT<int * (*)(), &func_intptr>();
  // CHECK: call {{.*}} @"??$FuncPtrAutoT@P6APEAHXZ$1?func_intptr@@YAPEAHXZ@@YAP6A?A_PXZXZ"
}

template<class T>
auto& RefAutoT(T& x) { return x; }

template<class T>
const auto& ConstRefAutoT(T& x) { return x; }

template<class T>
auto&& RRefAutoT(T& x) { return static_cast<int&&>(x); }

void test_template_ref_auto() {
  int x;

  RefAutoT(x);
  // CHECK: call {{.*}} @"??$RefAutoT@H@@YAAEA_PAEAH@Z"

  ConstRefAutoT(x);
  // CHECK: call {{.*}} @"??$ConstRefAutoT@H@@YAAEB_PAEAH@Z"

  RRefAutoT(x);
  // CHECK: call {{.*}} @"??$RRefAutoT@H@@YA$$QEA_PAEAH@Z"
}

template<class T>
decltype(auto) DecltypeAutoT() { return T(); }

template<class T>
decltype(auto) DecltypeAutoT2(T& x) { return static_cast<T&&>(x); }

void test_template_decltypeauto() {
  DecltypeAutoT<void>();
  // CHECK: call {{.*}} @"??$DecltypeAutoT@X@@YA?A_TXZ"

  DecltypeAutoT<const void>();
  // CHECK: call {{.*}} @"??$DecltypeAutoT@$$CBX@@YA?A_TXZ"

  DecltypeAutoT<volatile void>();
  // CHECK: call {{.*}} @"??$DecltypeAutoT@$$CCX@@YA?A_TXZ"

  DecltypeAutoT<const volatile void>();
  // CHECK: call {{.*}} @"??$DecltypeAutoT@$$CDX@@YA?A_TXZ"

  DecltypeAutoT<int>();
  // CHECK: call {{.*}} @"??$DecltypeAutoT@H@@YA?A_TXZ"

  DecltypeAutoT<const int>();
  // CHECK: call {{.*}} @"??$DecltypeAutoT@$$CBH@@YA?A_TXZ"

  DecltypeAutoT<volatile int>();
  // CHECK: call {{.*}} @"??$DecltypeAutoT@$$CCH@@YA?A_TXZ"

  DecltypeAutoT<const volatile int>();
  // CHECK: call {{.*}} @"??$DecltypeAutoT@$$CDH@@YA?A_TXZ"

  int x;

  DecltypeAutoT2(x);
  // CHECK: call {{.*}} @"??$DecltypeAutoT2@H@@YA?A_TAEAH@Z"
}

// Still want to use clang's custom mangling for lambdas to keep backwards compatibility until
// MSVC lambda name mangling has been deciphered.
void test_lambda() {
  int i = 0;

  auto lambdaIntRetAuto = []() { return 0; };
  lambdaIntRetAuto();
  // CHECK: call {{.*}} @"??R<lambda_1>@?0??test_lambda@@YAXXZ@QEBA?A?<auto>@@XZ"

  auto lambdaIntRet = []() -> int { return 0; };
  lambdaIntRet();
  // CHECK: call {{.*}} @"??R<lambda_2>@?0??test_lambda@@YAXXZ@QEBA@XZ"

  auto lambdaGenericIntIntRetAuto = [](auto a) { return a; };
  lambdaGenericIntIntRetAuto(0);
  // CHECK: call {{.*}} @"??$?RH@<lambda_0>@?0??test_lambda@@YAXXZ@QEBA?A?<auto>@@H@Z"

  auto lambdaRetTrailingAuto = []() -> auto { return 0; };
  lambdaRetTrailingAuto();
  // CHECK: call {{.*}} @"??R<lambda_3>@?0??test_lambda@@YAXXZ@QEBA?A?<auto>@@XZ"

  auto lambdaRetTrailingDecltypeAuto = []() -> decltype(auto) { return 0; };
  lambdaRetTrailingDecltypeAuto();
  // CHECK: call {{.*}} @"??R<lambda_4>@?0??test_lambda@@YAXXZ@QEBA?A?<decltype-auto>@@XZ"

  auto lambdaRetTrailingRefCollapse = [](int x) -> auto&& { return x; };
  lambdaRetTrailingRefCollapse(i);
  // CHECK: call {{.*}} @"??R<lambda_5>@?0??test_lambda@@YAXXZ@QEBA?A?<auto>@@H@Z"
}

auto TestTrailingInt() -> int {
  return 0;
}

auto TestTrailingConstVolatileVoid() -> const volatile void {
}

auto TestTrailingStructA() -> StructA {
  return StructA{};
}

void test_trailing_return() {
  TestTrailingInt();
  // CHECK: call {{.*}} @"?TestTrailingInt@@YAHXZ"

  TestTrailingConstVolatileVoid();
  // CHECK: call {{.*}} @"?TestTrailingConstVolatileVoid@@YAXXZ"

  TestTrailingStructA();
  // CHECK: call {{.*}} @"?TestTrailingStructA@@YA?AUStructA@@XZ"
}

auto TestNonTemplateAutoInt() {
  return 0;
}

auto TestNonTemplateAutoVoid() {
  return;
}

auto TestNonTemplateAutoStructA() {
  return StructA{};
}

const auto TestNonTemplateConstAutoInt() {
  return 0;
}

const auto TestNonTemplateConstAutoVoid() {
  return;
}

const auto TestNonTemplateConstAutoStructA() {
  return StructA{};
}

void test_nontemplate_auto() {
  TestNonTemplateAutoInt();
  // CHECK: call {{.*}} @"?TestNonTemplateAutoInt@@YA@XZ"

  TestNonTemplateAutoVoid();
  // CHECK: call {{.*}} @"?TestNonTemplateAutoVoid@@YA@XZ"

  TestNonTemplateAutoStructA();
  // CHECK: call {{.*}} @"?TestNonTemplateAutoStructA@@YA@XZ"

  TestNonTemplateConstAutoInt();
  // CHECK: call {{.*}} @"?TestNonTemplateConstAutoInt@@YA@XZ"

  TestNonTemplateConstAutoVoid();
  // CHECK: call {{.*}} @"?TestNonTemplateConstAutoVoid@@YA@XZ"

  TestNonTemplateConstAutoStructA();
  // CHECK: call {{.*}} @"?TestNonTemplateConstAutoStructA@@YA@XZ"
}

decltype(auto) TestNonTemplateDecltypeAutoInt() {
    return 0;
}

decltype(auto) TestNonTemplateDecltypeAutoVoid() {
    return;
}

decltype(auto) TestNonTemplateDecltypeAutoStructA() {
    return StructA{};
}

void test_nontemplate_decltypeauto() {
  TestNonTemplateDecltypeAutoInt();
  // CHECK: call {{.*}} @"?TestNonTemplateDecltypeAutoInt@@YA@XZ"

  TestNonTemplateDecltypeAutoVoid();
  // CHECK: call {{.*}} @"?TestNonTemplateDecltypeAutoVoid@@YA@XZ"

  TestNonTemplateDecltypeAutoStructA();
  // CHECK: call {{.*}} @"?TestNonTemplateDecltypeAutoStructA@@YA@XZ"
}

struct StructB {
  int x;
};

template<class T>
auto StructB::* AutoMemberDataPtrT(T x) { return x; }

template<class T>
const auto StructB::* AutoConstMemberDataPtrT(T x) { return x; }

void test_template_auto_member_data_ptr() {
  AutoMemberDataPtrT(&StructB::x);
  // CHECK: call {{.*}} @"??$AutoMemberDataPtrT@PEQStructB@@H@@YAPEQStructB@@_PPEQ0@H@Z"

  AutoConstMemberDataPtrT(&StructB::x);
  // CHECK: call {{.*}} @"??$AutoConstMemberDataPtrT@PEQStructB@@H@@YAPERStructB@@_PPEQ0@H@Z"
}

struct StructC {
  void test() {}
};

struct StructD {
  const int test() { return 0; }
};

template<class T>
auto (StructC::*AutoMemberFuncPtrT(T x))() { return x; }

template<class T>
const auto (StructD::*AutoConstMemberFuncPtrT(T x))() { return x; }

void test_template_auto_member_func_ptr() {
  AutoMemberFuncPtrT(&StructC::test);
  // CHECK: call {{.*}} @"??$AutoMemberFuncPtrT@P8StructC@@EAAXXZ@@YAP8StructC@@EAA?A_PXZP80@EAAXXZ@Z"

  AutoConstMemberFuncPtrT(&StructD::test);
  // CHECK: call {{.*}} @"??$AutoConstMemberFuncPtrT@P8StructD@@EAA?BHXZ@@YAP8StructD@@EAA?B_PXZP80@EAA?BHXZ@Z"
}

template<class T>
auto * __attribute__((address_space(1))) * AutoPtrAddressSpaceT() {
  T * __attribute__((address_space(1))) * p = nullptr;
  return p;
}

void test_template_auto_address_space_ptr() {
  AutoPtrAddressSpaceT<int>();
  // CHECK: call {{.*}} @"??$AutoPtrAddressSpaceT@H@@YA?A?<auto>@@XZ"
}

template<class T>
auto&& AutoReferenceCollapseT(T& x) { return static_cast<T&>(x); }

auto&& AutoReferenceCollapse(int& x) { return static_cast<int&>(x); }

void test2() {
  int x = 1;
  auto&& rref0 = AutoReferenceCollapseT(x);
  // CHECK: call {{.*}} @"??$AutoReferenceCollapseT@H@@YA$$QEA_PAEAH@Z"

  auto&& rref1 = AutoReferenceCollapse(x);
  // CHECK: call {{.*}} @"?AutoReferenceCollapse@@YA@AEAH@Z"
}
