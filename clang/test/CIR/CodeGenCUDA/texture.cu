
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

//===----------------------------------------------------------------------===//
// Device-side tests
//===----------------------------------------------------------------------===//

// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-cir -o - %s | FileCheck --check-prefix=CIR-DEVICE %s
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=LLVM-DEVICE %s
// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=OGCG-DEVICE %s

//===----------------------------------------------------------------------===//
// Host-side tests
//===----------------------------------------------------------------------===//

// RUN: echo -n "GPU binary would be here." > %t
// RUN: %clang_cc1 -fclangir -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t \
// RUN:   -emit-cir -o - %s | FileCheck --check-prefix=CIR-HOST %s
// RUN: %clang_cc1 -fclangir -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=LLVM-HOST %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=LLVM-HOST %s

struct textureReference {
  int desc;
};

enum ReadMode {
  ElementType = 0,
  NormalizedFloat = 1
};

template <typename T, int dim = 1, ReadMode mode = ElementType>
struct __attribute__((device_builtin_texture_type)) texture
    : public textureReference {};

texture<float, 2, NormalizedFloat> tex;
texture<float, 1, ElementType> tex_element;

//===----------------------------------------------------------------------===//
// Device-side checks
//===----------------------------------------------------------------------===//

// CIR-DEVICE: cir.global external target_address_space(1) @tex = #cir.undef : !cir.cuda_texture
// CIR-DEVICE: cir.global external target_address_space(1) @tex_element = #cir.undef : !cir.cuda_texture

// CIR now matches OG CodeGen and emits undef for CUDA shadow variables.
// LLVM-DEVICE: @tex ={{.*}} addrspace(1) externally_initialized global i64 undef
// LLVM-DEVICE: @tex_element ={{.*}} addrspace(1) externally_initialized global i64 undef

// OGCG-DEVICE: @tex ={{.*}} addrspace(1) externally_initialized global i64 undef
// OGCG-DEVICE: @tex_element ={{.*}} addrspace(1) externally_initialized global i64 undef

//===----------------------------------------------------------------------===//
// Host-side checks
//===----------------------------------------------------------------------===//

// Check the CUDA texture registration runtime declaration.
// CIR-HOST: cir.func private @__cudaRegisterTexture(!cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !s32i, !s32i)

// Check the CIR registration calls and their arguments.
// CIR-HOST-LABEL: cir.func internal private @__cuda_register_globals(
// CIR-HOST-SAME: %[[FATBIN:.*]]: !cir.ptr<!cir.ptr<!void>>

// First texture: texture_type = 2, normalized = 1.
// CIR-HOST: %[[TEX_GLOBAL:.*]] = cir.get_global @tex
// CIR-HOST: %[[TEX_PTR:.*]] = cir.cast bitcast %[[TEX_GLOBAL]]
// CIR-HOST: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST: %[[TEX_TYPE:.*]] = cir.const #cir.int<2> : !s32i
// CIR-HOST: %[[NORMALIZED:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST: cir.call @__cudaRegisterTexture(%[[FATBIN]], %[[TEX_PTR]], {{.*}}, {{.*}}, %[[TEX_TYPE]], %[[NORMALIZED]], %[[ZERO]])

// Second texture: texture_type = 1, normalized = 0.
// CIR-HOST: %[[ELEMENT_GLOBAL:.*]] = cir.get_global @tex_element
// CIR-HOST: %[[ELEMENT_PTR:.*]] = cir.cast bitcast %[[ELEMENT_GLOBAL]]
// CIR-HOST: %[[ELEMENT_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST: %[[ELEMENT_TYPE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST: %[[ELEMENT_NORMALIZED:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST: cir.call @__cudaRegisterTexture(%[[FATBIN]], %[[ELEMENT_PTR]], {{.*}}, {{.*}}, %[[ELEMENT_TYPE]], %[[ELEMENT_NORMALIZED]], %[[ELEMENT_ZERO]])

// Check texture registration metadata.
// CIR-HOST-DAG: @tex = {{.*}}cu.var_registration = #cir.cu.var_registration<tex, Texture, texture_type = 2, normalized>
// CIR-HOST-DAG: @tex_element = {{.*}}cu.var_registration = #cir.cu.var_registration<tex_element, Texture, texture_type = 1>

// Check that the host-side shadow variables are internalized.
// LLVM-HOST-DAG: @tex = internal global
// LLVM-HOST-DAG: @tex_element = internal global

// Check both CIR-lowered and original Clang CodeGen registration calls.
// LLVM-HOST-LABEL: define internal void @__cuda_register_globals
// LLVM-HOST-SAME: (ptr %[[FATBIN:.*]])
// LLVM-HOST: call void @__cudaRegisterTexture(ptr %[[FATBIN]], ptr @tex, ptr @[[TEX_NAME:.*]], ptr @[[TEX_NAME]], i32 2, i32 1, i32 0)
// LLVM-HOST: call void @__cudaRegisterTexture(ptr %[[FATBIN]], ptr @tex_element, ptr @[[ELEMENT_NAME:.*]], ptr @[[ELEMENT_NAME]], i32 1, i32 0, i32 0)
