// RUN: %clang_cc1 -std=cl2.0 -fapply-global-visibility-to-externs -fvisibility=default -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck --check-prefix=FVIS-DEFAULT %s
// RUN: %clang_cc1 -std=cl2.0 -fapply-global-visibility-to-externs -fvisibility=protected -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck --check-prefix=FVIS-PROTECTED %s
// RUN: %clang_cc1 -std=cl2.0 -fapply-global-visibility-to-externs -fvisibility=hidden -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck --check-prefix=FVIS-HIDDEN %s

// REQUIRES: amdgpu-registered-target

// FVIS-DEFAULT:  @glob ={{.*}} local_unnamed_addr addrspace(1) global i32 0
// FVIS-PROTECTED: @glob = protected local_unnamed_addr
// FVIS-HIDDEN: @glob = hidden local_unnamed_addr
int glob = 0;
// FVIS-DEFAULT:  @glob_hidden = hidden local_unnamed_addr
// FVIS-PROTECTED: @glob_hidden = hidden local_unnamed_addr
// FVIS-HIDDEN: @glob_hidden = hidden local_unnamed_addr
__attribute__((visibility("hidden"))) int glob_hidden = 0;
// FVIS-DEFAULT:  @glob_protected = protected local_unnamed_addr
// FVIS-PROTECTED: @glob_protected = protected local_unnamed_addr
// FVIS-HIDDEN: @glob_protected = protected local_unnamed_addr
__attribute__((visibility("protected"))) int glob_protected = 0;
// FVIS-DEFAULT:  @glob_default ={{( dso_local)?}} local_unnamed_addr
// FVIS-PROTECTED: @glob_default ={{( dso_local)?}} local_unnamed_addr
// FVIS-HIDDEN: @glob_default ={{( dso_local)?}} local_unnamed_addr
__attribute__((visibility("default"))) int glob_default = 0;

// FVIS-DEFAULT:  @ext = external local_unnamed_addr
// FVIS-PROTECTED: @ext = external protected local_unnamed_addr
// FVIS-HIDDEN: @ext = external hidden local_unnamed_addr
extern int ext;
// FVIS-DEFAULT:  @ext_hidden = external hidden local_unnamed_addr
// FVIS-PROTECTED: @ext_hidden = external hidden local_unnamed_addr
// FVIS-HIDDEN: @ext_hidden = external hidden local_unnamed_addr
__attribute__((visibility("hidden"))) extern int ext_hidden;
// FVIS-DEFAULT:  @ext_protected = external protected local_unnamed_addr
// FVIS-PROTECTED: @ext_protected = external protected local_unnamed_addr
// FVIS-HIDDEN: @ext_protected = external protected local_unnamed_addr
__attribute__((visibility("protected"))) extern int ext_protected;
// FVIS-DEFAULT:  @ext_default = external local_unnamed_addr
// FVIS-PROTECTED: @ext_default = external local_unnamed_addr
// FVIS-HIDDEN: @ext_default = external local_unnamed_addr
__attribute__((visibility("default"))) extern int ext_default;

// FVIS-DEFAULT: define{{.*}} amdgpu_kernel void @kern()
// FVIS-PROTECTED: define protected amdgpu_kernel void @kern()
// FVIS-HIDDEN: define protected amdgpu_kernel void @kern()
kernel void kern() {}
// FVIS-DEFAULT: define protected amdgpu_kernel void @kern_hidden()
// FVIS-PROTECTED: define protected amdgpu_kernel void @kern_hidden()
// FVIS-HIDDEN: define protected amdgpu_kernel void @kern_hidden()
__attribute__((visibility("hidden"))) kernel void kern_hidden() {}
// FVIS-DEFAULT: define protected amdgpu_kernel void @kern_protected()
// FVIS-PROTECTED: define protected amdgpu_kernel void @kern_protected()
// FVIS-HIDDEN: define protected amdgpu_kernel void @kern_protected()
__attribute__((visibility("protected"))) kernel void kern_protected() {}
// FVIS-DEFAULT: define{{.*}} amdgpu_kernel void @kern_default()
// FVIS-PROTECTED: define{{.*}} amdgpu_kernel void @kern_default()
// FVIS-HIDDEN: define{{.*}} amdgpu_kernel void @kern_default()
__attribute__((visibility("default"))) kernel void kern_default() {}

// FVIS-DEFAULT: define{{.*}} void @func()
// FVIS-PROTECTED: define protected void @func()
// FVIS-HIDDEN: define hidden void @func()
void func() {}
// FVIS-DEFAULT: define hidden void @func_hidden()
// FVIS-PROTECTED: define hidden void @func_hidden()
// FVIS-HIDDEN: define hidden void @func_hidden()
__attribute__((visibility("hidden"))) void func_hidden() {}
// FVIS-DEFAULT: define protected void @func_protected()
// FVIS-PROTECTED: define protected void @func_protected()
// FVIS-HIDDEN: define protected void @func_protected()
__attribute__((visibility("protected"))) void func_protected() {}
// FVIS-DEFAULT: define{{.*}} void @func_default()
// FVIS-PROTECTED: define{{.*}} void @func_default()
// FVIS-HIDDEN: define{{.*}} void @func_default()
__attribute__((visibility("default"))) void func_default() {}

extern kernel void ext_kern();
__attribute__((visibility("hidden"))) extern kernel void ext_kern_hidden();
__attribute__((visibility("protected"))) extern kernel void ext_kern_protected();
__attribute__((visibility("default"))) extern kernel void ext_kern_default();

extern void ext_func();
__attribute__((visibility("hidden"))) extern void ext_func_hidden();
__attribute__((visibility("protected"))) extern void ext_func_protected();
__attribute__((visibility("default"))) extern void ext_func_default();

void use() {
    glob = ext + ext_hidden + ext_protected + ext_default;
    ext_kern();
    ext_kern_hidden();
    ext_kern_protected();
    ext_kern_default();
    ext_func();
    ext_func_hidden();
    ext_func_protected();
    ext_func_default();
}

// FVIS-DEFAULT: declare amdgpu_kernel void @ext_kern()
// FVIS-PROTECTED: declare protected amdgpu_kernel void @ext_kern()
// FVIS-HIDDEN: declare protected amdgpu_kernel void @ext_kern()

// FVIS-DEFAULT: declare protected amdgpu_kernel void @ext_kern_hidden()
// FVIS-PROTECTED: declare protected amdgpu_kernel void @ext_kern_hidden()
// FVIS-HIDDEN: declare protected amdgpu_kernel void @ext_kern_hidden()

// FVIS-DEFAULT: declare protected amdgpu_kernel void @ext_kern_protected()
// FVIS-PROTECTED: declare protected amdgpu_kernel void @ext_kern_protected()
// FVIS-HIDDEN: declare protected amdgpu_kernel void @ext_kern_protected()

// FVIS-DEFAULT: declare amdgpu_kernel void @ext_kern_default()
// FVIS-PROTECTED: declare amdgpu_kernel void @ext_kern_default()
// FVIS-HIDDEN: declare amdgpu_kernel void @ext_kern_default()


// FVIS-DEFAULT: declare void @ext_func()
// FVIS-PROTECTED: declare protected void @ext_func()
// FVIS-HIDDEN: declare hidden void @ext_func()

// FVIS-DEFAULT: declare hidden void @ext_func_hidden()
// FVIS-PROTECTED: declare hidden void @ext_func_hidden()
// FVIS-HIDDEN: declare hidden void @ext_func_hidden()

// FVIS-DEFAULT: declare protected void @ext_func_protected()
// FVIS-PROTECTED: declare protected void @ext_func_protected()
// FVIS-HIDDEN: declare protected void @ext_func_protected()

// FVIS-DEFAULT: declare void @ext_func_default()
// FVIS-PROTECTED: declare void @ext_func_default()
// FVIS-HIDDEN: declare void @ext_func_default()
