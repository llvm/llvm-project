// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++11 -fvisibility hidden -fapply-global-visibility-to-externs \
// RUN:     -emit-llvm -o - %s | FileCheck -check-prefixes=HIPDEV %s
// RUN: %clang_cc1 -triple x86_64 -std=c++11 \
// RUN:     -emit-llvm -o - %s | FileCheck -check-prefixes=HIPHOST %s
// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++11 -fvisibility hidden -fapply-global-visibility-to-externs \
// RUN:     -O3 -emit-llvm -o - %s | FileCheck -check-prefixes=HIPDEVUNSED %s

struct textureReference {
  int a;
};

template <class T, int texType, int hipTextureReadMode>
struct texture : public textureReference {
texture() { a = 1; }
};

__attribute__((hip_pinned_shadow)) texture<float, 2, 1> tex;
// CUDADEV-NOT: @tex
// CUDAHOST-NOT: call i32 @__hipRegisterVar{{.*}}@tex
// HIPDEV: @tex = external addrspace(1) global %struct.texture
// HIPDEV-NOT: declare{{.*}}void @_ZN7textureIfLi2ELi1EEC1Ev
// HIPHOST:  define{{.*}}@_ZN7textureIfLi2ELi1EEC1Ev
// HIPHOST:  call i32 @__hipRegisterVar{{.*}}@tex{{.*}}i32 0, i32 4, i32 0, i32 0)
// HIPDEVUNSED: @tex = external addrspace(1) global %struct.texture
// HIPDEVUNSED-NOT: declare{{.*}}void @_ZN7textureIfLi2ELi1EEC1Ev
