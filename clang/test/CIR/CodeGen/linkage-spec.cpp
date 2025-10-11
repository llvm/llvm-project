// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

extern "C" void TopLevelC(){}
// CHECK: cir.func dso_local @TopLevelC() #cir.inline<never> {
extern "C++" void TopLevelCpp(){}
// CHECK: cir.func dso_local @_Z11TopLevelCppv() #cir.inline<never> {

extern "C++" {
  void ExternCppEmpty(){}
  // CHECK: cir.func dso_local @_Z14ExternCppEmptyv() #cir.inline<never> {
  extern "C" void ExternCpp_C(){}
  // CHECK: cir.func dso_local @ExternCpp_C() #cir.inline<never> {
  extern "C++" void ExternCpp_Cpp(){}
  // CHECK: cir.func dso_local @_Z13ExternCpp_Cppv() #cir.inline<never> {

  extern "C" {
  void ExternCpp_CEmpty(){}
  // CHECK: cir.func dso_local @ExternCpp_CEmpty() #cir.inline<never> {
  extern "C" void ExternCpp_C_C(){}
  // CHECK: cir.func dso_local @ExternCpp_C_C() #cir.inline<never> {
  extern "C++" void ExternCpp_C_Cpp(){}
  // CHECK: cir.func dso_local @_Z15ExternCpp_C_Cppv() #cir.inline<never> {
  }
}

extern "C" {
  void ExternCEmpty(){}
  // CHECK: cir.func dso_local @ExternCEmpty() #cir.inline<never> {
  extern "C" void ExternC_C(){}
  // CHECK: cir.func dso_local @ExternC_C() #cir.inline<never> {
  extern "C++" void ExternC_Cpp(){}
  // CHECK: cir.func dso_local @_Z11ExternC_Cppv() #cir.inline<never> {
  extern "C++" {
  void ExternC_CppEmpty(){}
  // CHECK: cir.func dso_local @_Z16ExternC_CppEmptyv() #cir.inline<never> {
  extern "C" void ExternC_Cpp_C(){}
  // CHECK: cir.func dso_local @ExternC_Cpp_C() #cir.inline<never> {
  extern "C++" void ExternC_Cpp_Cpp(){}
  // CHECK: cir.func dso_local @_Z15ExternC_Cpp_Cppv() #cir.inline<never> {
  }
}

