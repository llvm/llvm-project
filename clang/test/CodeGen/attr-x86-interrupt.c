// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,LINUX -DUWORD=i64 -DFRAME="[5 x i64]" -DALIGN=8
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,LINUX -DUWORD=i32 -DFRAME="[5 x i32]" -DALIGN=4
// RUN: %clang_cc1 -triple x86_64-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,WIN -DUWORD=i64 -DFRAME="[5 x i64]" -DALIGN=8
// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,WIN -DUWORD=i32 -DFRAME="[5 x i32]" -DALIGN=4
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnux32 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,LINUX -DUWORD=i64 -DFRAME="[5 x i64]" -DALIGN=8

#ifdef __x86_64__
typedef __UINT64_TYPE__ uword;
#else
typedef __UINT32_TYPE__ uword;
#endif

// LINUX: @llvm.compiler.used = appending global [8 x ptr] [ptr @foo7, ptr @foo8, ptr @foo_typedef, ptr @frame_void, ptr @frame_struct, ptr @frame_struct_ec, ptr @frame_func, ptr @frame_later], section "llvm.metadata"
// WIN: @llvm.used = appending global [8 x ptr] [ptr @foo7, ptr @foo8, ptr @foo_typedef, ptr @frame_void, ptr @frame_struct, ptr @frame_struct_ec, ptr @frame_func, ptr @frame_later], section "llvm.metadata"

// CHECK: define dso_local x86_intrcc void @foo7(ptr noundef byval(i32) align [[ALIGN]] %{{.+}}, [[UWORD]] noundef %{{.+}})
__attribute__((interrupt)) void foo7(int *a, uword b) {}
// CHECK: define dso_local x86_intrcc void @foo8(ptr noundef byval(i32) align [[ALIGN]] %{{.+}})
__attribute__((interrupt)) void foo8(int *a) {}
typedef int *int_ptr;
// CHECK: define dso_local x86_intrcc void @foo_typedef(ptr noundef byval(i32) align [[ALIGN]] %{{.+}})
__attribute__((interrupt)) void foo_typedef(int_ptr a) {}

// When the frame's type gives no size (void, an incomplete type as in GCC's
// documentation, or a function type), the byval type is the five stack slots
// the backend assigns to the frame.

struct interrupt_frame;

// CHECK: define dso_local x86_intrcc void @frame_void(ptr noundef byval([[FRAME]]) align [[ALIGN]] %{{.+}})
__attribute__((interrupt)) void frame_void(void *f) {}

// CHECK: define dso_local x86_intrcc void @frame_struct(ptr noundef byval([[FRAME]]) align [[ALIGN]] %{{.+}})
__attribute__((interrupt)) void frame_struct(struct interrupt_frame *f) {}

// CHECK: define dso_local x86_intrcc void @frame_struct_ec(ptr noundef byval([[FRAME]]) align [[ALIGN]] %{{.+}}, [[UWORD]] noundef %{{.+}})
__attribute__((interrupt)) void frame_struct_ec(struct interrupt_frame *f,
                                                uword ec) {}

// CHECK: define dso_local x86_intrcc void @frame_func(ptr noundef byval([[FRAME]]) align [[ALIGN]] %{{.+}})
__attribute__((interrupt)) void frame_func(void (*f)(void)) {}

// A type completed after the handler is still incomplete where the handler is
// emitted.
struct frame_later;
// CHECK: define dso_local x86_intrcc void @frame_later(ptr noundef byval([[FRAME]]) align [[ALIGN]] %{{.+}})
__attribute__((interrupt)) void frame_later(struct frame_later *f) {}
struct frame_later {
  uword ip;
};

// CHECK: "disable-tail-calls"="true"
// CHECK-NOT: "disable-tail-calls"="false"
