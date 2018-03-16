// RUN: %clang_cc1 -std=c++11 -triple i686-pc-win32 %s -fms-extensions -fno-rtti -emit-llvm -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CXX
// RUN: %clang_cc1 -x c -triple i686-pc-win32 %s -fms-extensions -fno-rtti -emit-llvm -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-C

// Similar to predefined-expr.cpp, but not as exhaustive, since it's basically
// equivalent to __PRETTY_FUNCTION__.

#ifdef __cplusplus
extern "C"
#endif
int printf(const char *, ...);

void funcNoProto() {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK-C:   @"??_C@_0BL@IHLLLCAO@void?5__cdecl?5funcNoProto?$CI?$CJ?$AA@" = linkonce_odr unnamed_addr constant [27 x i8] c"void __cdecl funcNoProto()\00"
// CHECK-CXX: @"??_C@_0BP@PJOECCJN@void?5__cdecl?5funcNoProto?$CIvoid?$CJ?$AA@" = linkonce_odr unnamed_addr constant [31 x i8] c"void __cdecl funcNoProto(void)\00"

void funcNoParams(void) {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK: @"??_C@_0CA@GBIDFNBN@void?5__cdecl?5funcNoParams?$CIvoid?$CJ?$AA@" = linkonce_odr unnamed_addr constant [32 x i8] c"void __cdecl funcNoParams(void)\00"

void freeFunc(int *p, char c) {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK: @"??_C@_0CD@KLGMNNL@void?5__cdecl?5freeFunc?$CIint?5?$CK?0?5cha@" = linkonce_odr unnamed_addr constant [{{.*}} x i8] c"void __cdecl freeFunc(int *, char)\00"

#ifdef __cplusplus
void funcVarargs(...) {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK-CXX: @"??_C@_0BO@BOBPLEKP@void?5__cdecl?5funcVarargs?$CI?4?4?4?$CJ?$AA@" = linkonce_odr unnamed_addr constant [30 x i8] c"void __cdecl funcVarargs(...)\00"

struct TopLevelClass {
  void topLevelMethod(int *, char);
};
void TopLevelClass::topLevelMethod(int *, char) {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK-CXX: @"??_C@_0DL@OBHNMDP@void?5__thiscall?5TopLevelClass?3?3t@" = linkonce_odr unnamed_addr constant [{{.*}} x i8] c"void __thiscall TopLevelClass::topLevelMethod(int *, char)\00"

namespace NS {
struct NamespacedClass {
  void namespacedMethod(int *, char);
};
void NamespacedClass::namespacedMethod(int *, char) {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK-CXX: @"??_C@_0ED@PFDKIEBA@void?5__thiscall?5NS?3?3NamespacedCl@" = linkonce_odr unnamed_addr constant [{{.*}} x i8] c"void __thiscall NS::NamespacedClass::namespacedMethod(int *, char)\00"
}
#endif
