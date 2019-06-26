// RUN: %clang_cc1 -fcuda-is-device -ast-dump -ast-dump-filter tex -x hip %s | FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -ast-dump -ast-dump-filter tex -x hip %s | FileCheck -strict-whitespace %s
struct textureReference {
  int a;
};

// CHECK: HIPPinnedShadowAttr
template <class T, int texType, int hipTextureReadMode>
struct texture : public textureReference {
texture() { a = 1; }
};

__attribute__((hip_pinned_shadow)) texture<float, 1, 1> tex;
