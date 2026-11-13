// Check that the OpenCL image feature/extension macros are only defined on
// AMDGPU targets with image instructions.

// RUN: %clang_cc1 -triple amdgpu-- -cl-std=CL2.0 -E -dM %s \
// RUN:   | FileCheck --check-prefix=IMAGES %s
// RUN: %clang_cc1 -triple amdgpu9.0a-- -cl-std=CL2.0 -E -dM %s \
// RUN:   | FileCheck --check-prefix=IMAGES %s
// RUN: %clang_cc1 -triple amdgpu11.00-- -cl-std=CL3.0 -E -dM %s \
// RUN:   | FileCheck --check-prefixes=IMAGES,IMAGES-CL30 %s
// RUN: %clang_cc1 -triple amdgpu9.42-- -cl-std=CL2.0 -E -dM %s \
// RUN:   | FileCheck --check-prefix=NO-IMAGES %s
// RUN: %clang_cc1 -triple amdgpu9.50-- -cl-std=CL2.0 -E -dM %s \
// RUN:   | FileCheck --check-prefix=NO-IMAGES %s
// RUN: %clang_cc1 -triple amdgpu12.50-- -cl-std=CL2.0 -E -dM %s \
// RUN:   | FileCheck --check-prefix=NO-IMAGES %s
// RUN: %clang_cc1 -triple amdgpu12.50-- -cl-std=CL3.0 -E -dM %s \
// RUN:   | FileCheck --check-prefix=NO-IMAGES %s

// IMAGES-DAG: #define __opencl_c_images 1
// IMAGES-DAG: #define __opencl_c_read_write_images 1
// IMAGES-DAG: #define cl_khr_3d_image_writes 1
// IMAGES-DAG: #define cl_khr_mipmap_image 1
// IMAGES-DAG: #define cl_khr_mipmap_image_writes 1
// IMAGES-CL30-DAG: #define __opencl_c_3d_image_writes 1

// NO-IMAGES-NOT: #define __opencl_c_3d_image_writes
// NO-IMAGES-NOT: #define __opencl_c_images
// NO-IMAGES-NOT: #define __opencl_c_read_write_images
// NO-IMAGES-NOT: #define cl_khr_3d_image_writes
// NO-IMAGES-NOT: #define cl_khr_mipmap_image
// NO-IMAGES-NOT: #define cl_khr_mipmap_image_writes
