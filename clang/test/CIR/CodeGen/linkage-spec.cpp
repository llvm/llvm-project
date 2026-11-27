// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

extern "C" void TopLevelC(){}
// CHECK: cir.func no_inline dso_local @TopLevelC()
extern "C++" void TopLevelCpp(){}
// CHECK: cir.func no_inline dso_local @_Z11TopLevelCppv()

extern "C++" {
  void ExternCppEmpty(){}
  // CHECK: cir.func no_inline dso_local @_Z14ExternCppEmptyv()
  extern "C" void ExternCpp_C(){}
  // CHECK: cir.func no_inline dso_local @ExternCpp_C()
  extern "C++" void ExternCpp_Cpp(){}
  // CHECK: cir.func no_inline dso_local @_Z13ExternCpp_Cppv()

  extern "C" {
  void ExternCpp_CEmpty(){}
  // CHECK: cir.func no_inline dso_local @ExternCpp_CEmpty()
  extern "C" void ExternCpp_C_C(){}
  // CHECK: cir.func no_inline dso_local @ExternCpp_C_C()
  extern "C++" void ExternCpp_C_Cpp(){}
  // CHECK: cir.func no_inline dso_local @_Z15ExternCpp_C_Cppv()
  }
}

extern "C" {
  void ExternCEmpty(){}
  // CHECK: cir.func no_inline dso_local @ExternCEmpty()
  extern "C" void ExternC_C(){}
  // CHECK: cir.func no_inline dso_local @ExternC_C()
  extern "C++" void ExternC_Cpp(){}
  // CHECK: cir.func no_inline dso_local @_Z11ExternC_Cppv()
  extern "C++" {
  void ExternC_CppEmpty(){}
  // CHECK: cir.func no_inline dso_local @_Z16ExternC_CppEmptyv()
  extern "C" void ExternC_Cpp_C(){}
  // CHECK: cir.func no_inline dso_local @ExternC_Cpp_C()
  extern "C++" void ExternC_Cpp_Cpp(){}
  // CHECK: cir.func no_inline dso_local @_Z15ExternC_Cpp_Cppv()
  }
}

