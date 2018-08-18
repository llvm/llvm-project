// RUN: %clang_cc1 %s -triple=i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s

int f();

// CHECK: $"?x@selectany_init@@3HA" = comdat any
// CHECK: $"?x@?$A@H@explicit_template_instantiation@@2HB" = comdat any
// CHECK: $"?x@?$A@H@implicit_template_instantiation@@2HB" = comdat any

namespace simple_init {
#pragma init_seg(compiler)
int x = f();
// CHECK: @"?x@simple_init@@3HA" = dso_local global i32 0, align 4
// CHECK: @__cxx_init_fn_ptr = private constant void ()* @"??__Ex@simple_init@@YAXXZ", section ".CRT$XCC"

#pragma init_seg(lib)
int y = f();
// CHECK: @"?y@simple_init@@3HA" = dso_local global i32 0, align 4
// CHECK: @__cxx_init_fn_ptr.1 = private constant void ()* @"??__Ey@simple_init@@YAXXZ", section ".CRT$XCL"

#pragma init_seg(user)
int z = f();
// CHECK: @"?z@simple_init@@3HA" = dso_local global i32 0, align 4
// No function pointer!  This one goes on @llvm.global_ctors.
}

#pragma init_seg(".asdf")

namespace internal_init {
namespace {
int x = f();
// CHECK: @"?x@?A0x{{[^@]*}}@internal_init@@3HA" = internal global i32 0, align 4
// CHECK: @__cxx_init_fn_ptr.2 = private constant void ()* @"??__Ex@?A0x{{[^@]*}}@internal_init@@YAXXZ", section ".asdf"
}
}

namespace selectany_init {
int __declspec(selectany) x = f();
// CHECK: @"?x@selectany_init@@3HA" = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @__cxx_init_fn_ptr.3 = private constant void ()* @"??__Ex@selectany_init@@YAXXZ", section ".asdf", comdat($"?x@selectany_init@@3HA")
}

namespace explicit_template_instantiation {
template <typename T> struct A { static const int x; };
template <typename T> const int A<T>::x = f();
template struct A<int>;
// CHECK: @"?x@?$A@H@explicit_template_instantiation@@2HB" = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @__cxx_init_fn_ptr.4 = private constant void ()* @"??__Ex@?$A@H@explicit_template_instantiation@@2HB@YAXXZ", section ".asdf", comdat($"?x@?$A@H@explicit_template_instantiation@@2HB")
}

namespace implicit_template_instantiation {
template <typename T> struct A { static const int x; };
template <typename T> const int A<T>::x = f();
int g() { return A<int>::x; }
// CHECK: @"?x@?$A@H@implicit_template_instantiation@@2HB" = linkonce_odr dso_local global i32 0, comdat, align 4
// CHECK: @__cxx_init_fn_ptr.5 = private constant void ()* @"??__Ex@?$A@H@implicit_template_instantiation@@2HB@YAXXZ", section ".asdf", comdat($"?x@?$A@H@implicit_template_instantiation@@2HB")
}

// ... and here's where we emitted user level ctors.
// CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }]
// CHECK: [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_pragma_init_seg.cpp, i8* null }]

// We have to mark everything used so we can survive globalopt, even through
// LTO.  There's no way LLVM could really understand if data in the .asdf
// section is really used or dead.
//
// CHECK: @llvm.used = appending global [6 x i8*]
// CHECK: [i8* bitcast (void ()** @__cxx_init_fn_ptr to i8*),
// CHECK: i8* bitcast (void ()** @__cxx_init_fn_ptr.1 to i8*),
// CHECK: i8* bitcast (void ()** @__cxx_init_fn_ptr.2 to i8*),
// CHECK: i8* bitcast (void ()** @__cxx_init_fn_ptr.3 to i8*),
// CHECK: i8* bitcast (void ()** @__cxx_init_fn_ptr.4 to i8*),
// CHECK: i8* bitcast (void ()** @__cxx_init_fn_ptr.5 to i8*)], section "llvm.metadata"
