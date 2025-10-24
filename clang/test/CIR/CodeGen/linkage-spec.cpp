// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

extern "C" void TopLevelC(){}
// CHECK: cir.func dso_local @TopLevelC() inline(never) {
extern "C++" void TopLevelCpp(){}
// CHECK: cir.func dso_local @_Z11TopLevelCppv() inline(never) {

extern "C++" {
  void ExternCppEmpty(){}
  // CHECK: cir.func dso_local @_Z14ExternCppEmptyv() inline(never) {
  extern "C" void ExternCpp_C(){}
  // CHECK: cir.func dso_local @ExternCpp_C() inline(never) {
  extern "C++" void ExternCpp_Cpp(){}
  // CHECK: cir.func dso_local @_Z13ExternCpp_Cppv() inline(never) {

  extern "C" {
  void ExternCpp_CEmpty(){}
  // CHECK: cir.func dso_local @ExternCpp_CEmpty() inline(never) {
  extern "C" void ExternCpp_C_C(){}
  // CHECK: cir.func dso_local @ExternCpp_C_C() inline(never) {
  extern "C++" void ExternCpp_C_Cpp(){}
  // CHECK: cir.func dso_local @_Z15ExternCpp_C_Cppv() inline(never) {
  }
}

extern "C" {
  void ExternCEmpty(){}
  // CHECK: cir.func dso_local @ExternCEmpty() inline(never) {
  extern "C" void ExternC_C(){}
  // CHECK: cir.func dso_local @ExternC_C() inline(never) {
  extern "C++" void ExternC_Cpp(){}
  // CHECK: cir.func dso_local @_Z11ExternC_Cppv() inline(never) {
  extern "C++" {
  void ExternC_CppEmpty(){}
  // CHECK: cir.func dso_local @_Z16ExternC_CppEmptyv() inline(never) {
  extern "C" void ExternC_Cpp_C(){}
  // CHECK: cir.func dso_local @ExternC_Cpp_C() inline(never) {
  extern "C++" void ExternC_Cpp_Cpp(){}
  // CHECK: cir.func dso_local @_Z15ExternC_Cpp_Cppv() inline(never) {
  }
}

