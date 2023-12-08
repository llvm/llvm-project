// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fexceptions %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fexceptions %s -o - -DHIDDEN | FileCheck %s --check-prefixes CHECK,CHECK-HIDDEN

class A {
public:
  ~A();
} a;

// CHECK-DEFAULT: @__dso_handle = global ptr @__dso_handle, align 8
// CHECK-HIDDEN: @__dso_handle = hidden global ptr @__dso_handle, align 8
// CHECK: define internal void @__cxx_global_var_init()
// CHECK:   call i32 @__cxa_atexit({{.*}}, {{.*}}, ptr @__dso_handle)

#ifdef HIDDEN
void *__dso_handle __attribute__((__visibility__("hidden"))) = &__dso_handle;
#else
void *__dso_handle = &__dso_handle;
#endif

void use(void *);
void use_dso_handle() {
  use(__dso_handle);
}
