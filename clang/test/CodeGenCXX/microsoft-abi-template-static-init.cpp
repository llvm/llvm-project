// RUN: %clang_cc1 %s -triple=i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=i686-pc-windows-msvc -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-pc-windows-msvc -fms-extensions -emit-llvm -o - | FileCheck %s

struct S {
  S();
  ~S();
};

template <typename T> struct __declspec(dllexport) ExportedTemplate {
  static S s;
};
template <typename T> S ExportedTemplate<T>::s;
void useExportedTemplate(ExportedTemplate<int> x) {
  (void)x.s;
}
int f();
namespace selectany_init {
// MS don't put selectany static var in the linker directive, init routine
// f() is not getting called if x is not referenced.
int __declspec(selectany) x = f();
}

namespace explicit_template_instantiation {
template <typename T> struct A { static  int x; };
template <typename T> int A<T>::x = f();
template struct A<int>;
}

namespace implicit_template_instantiation {
template <typename T> struct A { static  int x; };
template <typename T>  int A<T>::x = f();
int g() { return A<int>::x; }
}


template <class T>
struct X_ {
  static T ioo;
  static T init();
};
template <class T> T X_<T>::ioo = X_<T>::init();
template struct X_<int>;

template <class T>
struct X {
  static T ioo;
  static T init();
};
// template specialized static data don't need in llvm.used,
// the static init routine get call from _GLOBAL__sub_I_ routines.
template <> int X<int>::ioo = X<int>::init();
template struct X<int>;
// CHECK: @llvm.global_ctors = appending global [6 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @"??__Ex@selectany_init@@YAXXZ", i8* bitcast (i32* @"?x@selectany_init@@3HA" to i8*) }, { i32, void ()*, i8* } { i32 65535, void ()* @"??__E?x@?$A@H@explicit_template_instantiation@@2HA@@YAXXZ", i8* bitcast (i32* @"?x@?$A@H@explicit_template_instantiation@@2HA" to i8*) }, { i32, void ()*, i8* } { i32 65535, void ()* @"??__E?ioo@?$X_@H@@2HA@@YAXXZ", i8* bitcast (i32* @"?ioo@?$X_@H@@2HA" to i8*) }, { i32, void ()*, i8* } { i32 65535, void ()* @"??__E?s@?$ExportedTemplate@H@@2US@@A@@YAXXZ", i8* getelementptr inbounds (%struct.S, %struct.S* @"?s@?$ExportedTemplate@H@@2US@@A", i32 0, i32 0) }, { i32, void ()*, i8* } { i32 65535, void ()* @"??__E?x@?$A@H@implicit_template_instantiation@@2HA@@YAXXZ", i8* bitcast (i32* @"?x@?$A@H@implicit_template_instantiation@@2HA" to i8*) }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_microsoft_abi_template_static_init.cpp, i8* null }]
// CHECK: @llvm.used = appending global [4 x i8*] [i8* bitcast (i32* @"?x@?$A@H@explicit_template_instantiation@@2HA" to i8*), i8* bitcast (i32* @"?ioo@?$X_@H@@2HA" to i8*), i8* getelementptr inbounds (%struct.S, %struct.S* @"?s@?$ExportedTemplate@H@@2US@@A", i32 0, i32 0), i8* bitcast (i32* @"?x@?$A@H@implicit_template_instantiation@@2HA" to i8*)], section "llvm.metadata"

