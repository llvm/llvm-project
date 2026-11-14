// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,X86_64
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,X86

// The interrupt frame's type may be unsized: an incomplete type, as in GCC's
// documentation, or a function type. byval needs a sized type, so i8 is used,
// as for "void *".

#ifdef __x86_64__
typedef __UINT64_TYPE__ uword;
#else
typedef __UINT32_TYPE__ uword;
#endif

struct interrupt_frame;
union frame_union;
typedef struct frame_tag frame_t;

// CHECK: define{{.*}} x86_intrcc void @frame_void(ptr noundef byval(i8) %{{.+}})
__attribute__((interrupt)) void frame_void(void *f) {}

// CHECK: define{{.*}} x86_intrcc void @frame_struct(ptr noundef byval(i8) %{{.+}})
__attribute__((interrupt)) void frame_struct(struct interrupt_frame *f) {}

// X86_64: define{{.*}} x86_intrcc void @frame_struct_ec(ptr noundef byval(i8) %{{.+}}, i64 noundef %{{.+}})
// X86: define{{.*}} x86_intrcc void @frame_struct_ec(ptr noundef byval(i8) %{{.+}}, i32 noundef %{{.+}})
__attribute__((interrupt)) void frame_struct_ec(struct interrupt_frame *f,
                                                uword ec) {}

// CHECK: define{{.*}} x86_intrcc void @frame_union(ptr noundef byval(i8) %{{.+}})
__attribute__((interrupt)) void frame_union(union frame_union *f) {}

// CHECK: define{{.*}} x86_intrcc void @frame_typedef(ptr noundef byval(i8) %{{.+}})
__attribute__((interrupt)) void frame_typedef(frame_t *f) {}

// CHECK: define{{.*}} x86_intrcc void @frame_func(ptr noundef byval(i8) %{{.+}})
__attribute__((interrupt)) void frame_func(void (*f)(void)) {}

// A type completed after the handler is still incomplete where the handler is
// emitted.
struct frame_later;
// CHECK: define{{.*}} x86_intrcc void @frame_later(ptr noundef byval(i8) %{{.+}})
__attribute__((interrupt)) void frame_later(struct frame_later *f) {}
struct frame_later {
  uword ip;
};
