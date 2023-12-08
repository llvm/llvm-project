// RUN: %clang_cc1 %s -triple=i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s

int f();

// CHECK: $"?x@selectany_init@@3HA" = comdat any
// CHECK: $"?x@?$A@H@explicit_template_instantiation@@2HB" = comdat any
// CHECK: $"?x@?$A@H@implicit_template_instantiation@@2HB" = comdat any

namespace simple_init {
#pragma init_seg(compiler)
int x = f();
// CHECK: @"?x@simple_init@@3HA" = dso_local global i32 0, align 4
// No function pointer!  This one goes on @llvm.global_ctors.

#pragma init_seg(lib)
int y = f();
// CHECK: @"?y@simple_init@@3HA" = dso_local global i32 0, align 4
// No function pointer!  This one goes on @llvm.global_ctors.

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
// CHECK: @__cxx_init_fn_ptr = private constant ptr @"??__Ex@?A0x{{[^@]*}}@internal_init@@YAXXZ", section ".asdf"
}
}

namespace selectany_init {
int __declspec(selectany) x = f();
// CHECK: @"?x@selectany_init@@3HA" = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @__cxx_init_fn_ptr.1 = private constant ptr @"??__Ex@selectany_init@@YAXXZ", section ".asdf", comdat($"?x@selectany_init@@3HA")
}

namespace explicit_template_instantiation {
template <typename T> struct A { static const int x; };
template <typename T> const int A<T>::x = f();
template struct A<int>;
// CHECK: @"?x@?$A@H@explicit_template_instantiation@@2HB" = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @__cxx_init_fn_ptr.2 = private constant ptr @"??__E?x@?$A@H@explicit_template_instantiation@@2HB@@YAXXZ", section ".asdf", comdat($"?x@?$A@H@explicit_template_instantiation@@2HB")
}

namespace implicit_template_instantiation {
template <typename T> struct A { static const int x; };
template <typename T> const int A<T>::x = f();
int g() { return A<int>::x; }
// CHECK: @"?x@?$A@H@implicit_template_instantiation@@2HB" = linkonce_odr dso_local global i32 0, comdat, align 4
// CHECK: @__cxx_init_fn_ptr.3 = private constant ptr @"??__E?x@?$A@H@implicit_template_instantiation@@2HB@@YAXXZ", section ".asdf", comdat($"?x@?$A@H@implicit_template_instantiation@@2HB")
}

// ... and here's where we emitted user level ctors.
// CHECK: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }]
// CHECK: [{ i32, ptr, ptr } { i32 200, ptr @"??__Ex@simple_init@@YAXXZ", ptr @"?x@simple_init@@3HA" }, { i32, ptr, ptr } { i32 400, ptr @"??__Ey@simple_init@@YAXXZ", ptr @"?y@simple_init@@3HA" }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_pragma_init_seg.cpp, ptr null }]

// We have to mark everything used so we can survive globalopt, even through
// LTO.  There's no way LLVM could really understand if data in the .asdf
// section is really used or dead.
//
// CHECK: @llvm.used = appending global [4 x ptr]
// CHECK: [ptr @__cxx_init_fn_ptr,
// CHECK: ptr @__cxx_init_fn_ptr.1,
// CHECK: ptr @__cxx_init_fn_ptr.2,
// CHECK: ptr @__cxx_init_fn_ptr.3], section "llvm.metadata"
