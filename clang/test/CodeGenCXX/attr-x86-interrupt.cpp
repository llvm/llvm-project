// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=X86_64_LINUX
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=X86_LINUX
// RUN: %clang_cc1 -triple x86_64-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefix=X86_64_WIN
// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefix=X86_WIN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnux32 %s -emit-llvm -o - | FileCheck %s --check-prefix=X86_64_LINUX

#ifdef __x86_64__
typedef __UINT64_TYPE__ uword;
#else
typedef __UINT32_TYPE__ uword;
#endif

__attribute__((interrupt)) void foo7(int *a, uword b) {}
namespace S {
__attribute__((interrupt)) void foo8(int *a) {}
}
struct St {
static void foo9(int *a) __attribute__((interrupt)) {}
};
// X86_64_LINUX: @llvm.compiler.used = appending global [3 x i8*] [i8* bitcast (void (i32*, i64)* @{{.*}}foo7{{.*}} to i8*), i8* bitcast (void (i32*)* @{{.*}}foo8{{.*}} to i8*), i8* bitcast (void (i32*)* @{{.*}}foo9{{.*}} to i8*)], section "llvm.metadata"
// X86_64_LINUX: define{{.*}} x86_intrcc void @{{.*}}foo7{{.*}}(i32* noundef byval(i32) %{{.+}}, i64 noundef %{{.+}})
// X86_64_LINUX: define{{.*}} x86_intrcc void @{{.*}}foo8{{.*}}(i32* noundef byval(i32) %{{.+}})
// X86_64_LINUX: define linkonce_odr x86_intrcc void @{{.*}}foo9{{.*}}(i32* noundef byval(i32) %{{.+}})
// X86_LINUX: @llvm.compiler.used = appending global [3 x i8*] [i8* bitcast (void (i32*, i32)* @{{.*}}foo7{{.*}} to i8*), i8* bitcast (void (i32*)* @{{.*}}foo8{{.*}} to i8*), i8* bitcast (void (i32*)* @{{.*}}foo9{{.*}} to i8*)], section "llvm.metadata"
// X86_LINUX: define{{.*}} x86_intrcc void @{{.*}}foo7{{.*}}(i32* noundef byval(i32) %{{.+}}, i32 noundef %{{.+}})
// X86_LINUX: define{{.*}} x86_intrcc void @{{.*}}foo8{{.*}}(i32* noundef byval(i32) %{{.+}})
// X86_LINUX: define linkonce_odr x86_intrcc void @{{.*}}foo9{{.*}}(i32* noundef byval(i32) %{{.+}})
// X86_64_WIN: @llvm.used = appending global [3 x i8*] [i8* bitcast (void (i32*, i64)* @{{.*}}foo7{{.*}} to i8*), i8* bitcast (void (i32*)* @{{.*}}foo8{{.*}} to i8*), i8* bitcast (void (i32*)* @{{.*}}foo9{{.*}} to i8*)], section "llvm.metadata"
// X86_64_WIN: define dso_local x86_intrcc void @{{.*}}foo7{{.*}}(i32* noundef byval(i32) %{{.+}}, i64 noundef %{{.+}})
// X86_64_WIN: define dso_local x86_intrcc void @{{.*}}foo8{{.*}}(i32* noundef byval(i32) %{{.+}})
// X86_64_WIN: define linkonce_odr dso_local x86_intrcc void @{{.*}}foo9{{.*}}(i32* noundef byval(i32) %{{.+}})
// X86_WIN: @llvm.used = appending global [3 x i8*] [i8* bitcast (void (i32*, i32)* @{{.*}}foo7{{.*}} to i8*), i8* bitcast (void (i32*)* @{{.*}}foo8{{.*}} to i8*), i8* bitcast (void (i32*)* @{{.*}}foo9{{.*}} to i8*)], section "llvm.metadata"
// X86_WIN: define dso_local x86_intrcc void @{{.*}}foo7{{.*}}(i32* noundef byval(i32) %{{.+}}, i32 noundef %{{.+}})
// X86_WIN: define dso_local x86_intrcc void @{{.*}}foo8{{.*}}(i32* noundef byval(i32) %{{.+}})
// X86_WIN: define linkonce_odr dso_local x86_intrcc void @{{.*}}foo9{{.*}}(i32* noundef byval(i32) %{{.+}})
