// Builtin bitcode is built without a processor and may enable image features on
// its own. On a processor without images the linked copy follows the processor;
// otherwise, and for other features, the library's features are kept.

// RUN: %clang_cc1 -triple amdgpu-- -DBITCODE -emit-llvm-bc -o %t-lib.bc %s

// RUN: %clang_cc1 -triple amdgpu12.50-- -emit-llvm -o - %s \
// RUN:   -mlink-builtin-bitcode %t-lib.bc | FileCheck %s \
// RUN:   --check-prefixes=NOIMAGE,GWS --implicit-check-not=+image-insts \
// RUN:   --implicit-check-not=+extended-image-insts
// RUN: %clang_cc1 -triple amdgpu9.42-- -emit-llvm -o - %s \
// RUN:   -mlink-builtin-bitcode %t-lib.bc | FileCheck %s \
// RUN:   --check-prefix=NOIMAGE --implicit-check-not=+image-insts \
// RUN:   --implicit-check-not=+extended-image-insts
// RUN: %clang_cc1 -triple amdgpu12.50-- -emit-llvm -o - %s \
// RUN:   -mlink-builtin-bitcode-postopt -mlink-builtin-bitcode %t-lib.bc \
// RUN:   | FileCheck %s --check-prefixes=NOIMAGE,GWS \
// RUN:   --implicit-check-not=+image-insts \
// RUN:   --implicit-check-not=+extended-image-insts
// RUN: %clang_cc1 -triple amdgpu9.0a-- -emit-llvm -o - %s \
// RUN:   -mlink-builtin-bitcode %t-lib.bc | FileCheck %s --check-prefix=IMAGE
// RUN: %clang_cc1 -triple amdgpu-- -emit-llvm -o - %s \
// RUN:   -mlink-builtin-bitcode %t-lib.bc | FileCheck %s --check-prefix=IMAGE
// RUN: %clang_cc1 -triple r600-- -DBITCODE -emit-llvm-bc -o %t-r600-lib.bc %s
// RUN: %clang_cc1 -triple r600-- -target-cpu cypress -emit-llvm -o - %s \
// RUN:   -mlink-builtin-bitcode %t-r600-lib.bc \
// RUN:   | FileCheck %s --check-prefix=IMAGE
// RUN: %if cir-enabled %{ %clang_cc1 -triple amdgpu12.50-- -fclangir \
// RUN:   -emit-llvm -o - %s -mlink-builtin-bitcode %t-lib.bc | FileCheck %s \
// RUN:   --check-prefixes=NOIMAGE,GWS --implicit-check-not=+image-insts \
// RUN:   --implicit-check-not=+extended-image-insts %}

// The linked image operation is then diagnosed instead of being selected.
// RUN: %clang_cc1 -triple amdgpu-- -DBITCODE -DIMAGE_OP -emit-llvm-bc \
// RUN:   -o %t-op.bc %s
// RUN: %if amdgpu-registered-target %{ not %clang_cc1 -triple amdgpu12.50-- \
// RUN:   -DIMAGE_OP -S -o /dev/null %s -mlink-builtin-bitcode %t-op.bc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERR %}

#ifdef IMAGE_OP
typedef float float4 __attribute__((ext_vector_type(4)));
#ifdef BITCODE
__attribute__((target("image-insts")))
float4 image_op(int x, __amdgpu_texture_t tex) {
  return __builtin_amdgcn_image_load_2d_v4f32_i32(15, x, x, tex, 0, 0);
}
#else
float4 image_op(int x, __amdgpu_texture_t tex);
float4 foo(int x, __amdgpu_texture_t tex) { return image_op(x, tex); }
#endif
#elif defined(BITCODE)
__attribute__((target("image-insts,16-bit-insts")))
int image_fn(void) { return 1; }
__attribute__((target("extended-image-insts"))) int ext_fn(void) { return 2; }
__attribute__((target("gws"))) int gws_fn(void) { return 3; }
#else
int image_fn(void);
int ext_fn(void);
int gws_fn(void);
int foo(void) { return image_fn() + ext_fn() + gws_fn(); }
#endif

// ERR: error: llvm.amdgcn.image.load.2d requires target feature 'image-insts'

// NOIMAGE-LABEL: define internal i32 @image_fn(
// NOIMAGE-SAME: #[[CPU:[0-9]+]]
// NOIMAGE-LABEL: define internal i32 @ext_fn(
// NOIMAGE-SAME: #[[CPU]]
// GWS-LABEL: define internal i32 @gws_fn(
// GWS-SAME: #[[GWS:[0-9]+]]
// NOIMAGE: attributes #[[CPU]] = { {{.*}}"target-features"="{{.*}}+gfx9-insts
// GWS: attributes #[[GWS]] = { {{.*}}"target-features"="{{.*}}+gws

// IMAGE-LABEL: define internal i32 @image_fn(
// IMAGE-SAME: #[[IMAGE:[0-9]+]]
// IMAGE-LABEL: define internal i32 @ext_fn(
// IMAGE-SAME: #[[EXT:[0-9]+]]
// IMAGE: attributes #[[IMAGE]] = { {{.*}}"target-features"="{{.*}}+image-insts
// IMAGE: attributes #[[EXT]] = { {{.*}}"target-features"="{{.*}}+extended-image-insts
