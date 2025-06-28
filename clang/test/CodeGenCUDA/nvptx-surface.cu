// RUN: %clang_cc1 -triple nvptx-unknown-unknown -fcuda-is-device -O3 -o - %s -emit-llvm | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -fcuda-is-device -O3 -o - %s -emit-llvm | FileCheck %s
#include "Inputs/cuda.h"

#include "__clang_cuda_texture_intrinsics.h"

__device__ void surfchar(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  char val;

  // CHECK: %0 = tail call i8 asm "suld.b.1d.b8.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i8 asm "suld.b.1d.b8.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i8 asm "suld.b.1d.b8.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i8 asm "suld.b.2d.b8.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i8 asm "suld.b.2d.b8.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i8 asm "suld.b.2d.b8.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i8 asm "suld.b.3d.b8.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i8 asm "suld.b.3d.b8.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i8 asm "suld.b.3d.b8.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i8 asm "suld.b.a1d.b8.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i8 asm "suld.b.a1d.b8.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i8 asm "suld.b.a1d.b8.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfsignedchar(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  signed char val;

  // CHECK: %0 = tail call i8 asm "suld.b.1d.b8.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i8 asm "suld.b.1d.b8.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i8 asm "suld.b.1d.b8.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i8 asm "suld.b.2d.b8.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i8 asm "suld.b.2d.b8.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i8 asm "suld.b.2d.b8.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i8 asm "suld.b.3d.b8.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i8 asm "suld.b.3d.b8.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i8 asm "suld.b.3d.b8.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i8 asm "suld.b.a1d.b8.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i8 asm "suld.b.a1d.b8.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i8 asm "suld.b.a1d.b8.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfchar1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  char1 val;

  // CHECK: %0 = tail call i8 asm "suld.b.1d.b8.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i8 asm "suld.b.1d.b8.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i8 asm "suld.b.1d.b8.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i8 asm "suld.b.2d.b8.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i8 asm "suld.b.2d.b8.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i8 asm "suld.b.2d.b8.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i8 asm "suld.b.3d.b8.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i8 asm "suld.b.3d.b8.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i8 asm "suld.b.3d.b8.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i8 asm "suld.b.a1d.b8.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i8 asm "suld.b.a1d.b8.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i8 asm "suld.b.a1d.b8.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfunsignedchar(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  unsigned char val;

  // CHECK: %0 = tail call i8 asm "suld.b.1d.b8.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i8 asm "suld.b.1d.b8.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i8 asm "suld.b.1d.b8.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i8 asm "suld.b.2d.b8.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i8 asm "suld.b.2d.b8.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i8 asm "suld.b.2d.b8.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i8 asm "suld.b.3d.b8.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i8 asm "suld.b.3d.b8.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i8 asm "suld.b.3d.b8.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i8 asm "suld.b.a1d.b8.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i8 asm "suld.b.a1d.b8.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i8 asm "suld.b.a1d.b8.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfuchar1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  uchar1 val;

  // CHECK: %0 = tail call i8 asm "suld.b.1d.b8.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i8 asm "suld.b.1d.b8.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i8 asm "suld.b.1d.b8.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b8.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i8 asm "suld.b.2d.b8.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i8 asm "suld.b.2d.b8.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i8 asm "suld.b.2d.b8.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b8.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i8 asm "suld.b.3d.b8.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i8 asm "suld.b.3d.b8.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i8 asm "suld.b.3d.b8.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b8.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i8 asm "suld.b.a1d.b8.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i8 asm "suld.b.a1d.b8.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i8 asm "suld.b.a1d.b8.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b8.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i8 asm "suld.b.a2d.b8.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i8 asm "suld.b.a2d.b8.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i8 asm "suld.b.a2d.b8.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b8.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfshort(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  short val;

  // CHECK: %0 = tail call i16 asm "suld.b.1d.b16.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i16 asm "suld.b.1d.b16.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i16 asm "suld.b.1d.b16.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i16 asm "suld.b.2d.b16.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i16 asm "suld.b.2d.b16.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i16 asm "suld.b.2d.b16.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i16 asm "suld.b.3d.b16.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i16 asm "suld.b.3d.b16.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i16 asm "suld.b.3d.b16.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i16 asm "suld.b.a1d.b16.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i16 asm "suld.b.a1d.b16.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i16 asm "suld.b.a1d.b16.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfshort1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  short1 val;

  // CHECK: %0 = tail call i16 asm "suld.b.1d.b16.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i16 asm "suld.b.1d.b16.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i16 asm "suld.b.1d.b16.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i16 asm "suld.b.2d.b16.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i16 asm "suld.b.2d.b16.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i16 asm "suld.b.2d.b16.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i16 asm "suld.b.3d.b16.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i16 asm "suld.b.3d.b16.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i16 asm "suld.b.3d.b16.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i16 asm "suld.b.a1d.b16.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i16 asm "suld.b.a1d.b16.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i16 asm "suld.b.a1d.b16.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfunsignedshort(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  unsigned short val;

  // CHECK: %0 = tail call i16 asm "suld.b.1d.b16.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i16 asm "suld.b.1d.b16.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i16 asm "suld.b.1d.b16.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i16 asm "suld.b.2d.b16.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i16 asm "suld.b.2d.b16.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i16 asm "suld.b.2d.b16.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i16 asm "suld.b.3d.b16.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i16 asm "suld.b.3d.b16.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i16 asm "suld.b.3d.b16.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i16 asm "suld.b.a1d.b16.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i16 asm "suld.b.a1d.b16.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i16 asm "suld.b.a1d.b16.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfushort1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  ushort1 val;

  // CHECK: %0 = tail call i16 asm "suld.b.1d.b16.zero {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.zero [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i16 asm "suld.b.1d.b16.clamp {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.clamp [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i16 asm "suld.b.1d.b16.trap {$0}, [$1, {$2}];", "=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b16.trap [$0, {$1}], {$2};", "l,r,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i16 asm "suld.b.2d.b16.zero {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.zero [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i16 asm "suld.b.2d.b16.clamp {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.clamp [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i16 asm "suld.b.2d.b16.trap {$0}, [$1, {$2, $3}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b16.trap [$0, {$1, $2}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i16 asm "suld.b.3d.b16.zero {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i16 asm "suld.b.3d.b16.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i16 asm "suld.b.3d.b16.trap {$0}, [$1, {$2, $3, $4, $4}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b16.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i16 asm "suld.b.a1d.b16.zero {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.zero [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i16 asm "suld.b.a1d.b16.clamp {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.clamp [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i16 asm "suld.b.a1d.b16.trap {$0}, [$1, {$3, $2}];", "=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b16.trap [$0, {$2, $1}], {$3};", "l,r,r,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i16 asm "suld.b.a2d.b16.zero {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i16 asm "suld.b.a2d.b16.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i16 asm "suld.b.a2d.b16.trap {$0}, [$1, {$4, $2, $3, $3}];", "=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b16.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfint(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  int val;

  // CHECK: %0 = tail call i32 asm "suld.b.1d.b32.zero {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.zero [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i32 asm "suld.b.1d.b32.clamp {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.clamp [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i32 asm "suld.b.1d.b32.trap {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.trap [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i32 asm "suld.b.2d.b32.zero {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.zero [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i32 asm "suld.b.2d.b32.clamp {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.clamp [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i32 asm "suld.b.2d.b32.trap {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.trap [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i32 asm "suld.b.3d.b32.zero {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i32 asm "suld.b.3d.b32.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i32 asm "suld.b.3d.b32.trap {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i32 asm "suld.b.a1d.b32.zero {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.zero [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i32 asm "suld.b.a1d.b32.clamp {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.clamp [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i32 asm "suld.b.a1d.b32.trap {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.trap [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfint1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  int1 val;

  // CHECK: %0 = tail call i32 asm "suld.b.1d.b32.zero {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.zero [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i32 asm "suld.b.1d.b32.clamp {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.clamp [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i32 asm "suld.b.1d.b32.trap {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.trap [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i32 asm "suld.b.2d.b32.zero {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.zero [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i32 asm "suld.b.2d.b32.clamp {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.clamp [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i32 asm "suld.b.2d.b32.trap {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.trap [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i32 asm "suld.b.3d.b32.zero {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i32 asm "suld.b.3d.b32.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i32 asm "suld.b.3d.b32.trap {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i32 asm "suld.b.a1d.b32.zero {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.zero [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i32 asm "suld.b.a1d.b32.clamp {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.clamp [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i32 asm "suld.b.a1d.b32.trap {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.trap [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfunsignedint(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  unsigned int val;

  // CHECK: %0 = tail call i32 asm "suld.b.1d.b32.zero {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.zero [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i32 asm "suld.b.1d.b32.clamp {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.clamp [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i32 asm "suld.b.1d.b32.trap {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.trap [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i32 asm "suld.b.2d.b32.zero {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.zero [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i32 asm "suld.b.2d.b32.clamp {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.clamp [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i32 asm "suld.b.2d.b32.trap {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.trap [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i32 asm "suld.b.3d.b32.zero {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i32 asm "suld.b.3d.b32.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i32 asm "suld.b.3d.b32.trap {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i32 asm "suld.b.a1d.b32.zero {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.zero [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i32 asm "suld.b.a1d.b32.clamp {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.clamp [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i32 asm "suld.b.a1d.b32.trap {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.trap [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfuint1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  uint1 val;

  // CHECK: %0 = tail call i32 asm "suld.b.1d.b32.zero {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.zero [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i32 asm "suld.b.1d.b32.clamp {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.clamp [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i32 asm "suld.b.1d.b32.trap {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.trap [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i32 asm "suld.b.2d.b32.zero {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.zero [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i32 asm "suld.b.2d.b32.clamp {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.clamp [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i32 asm "suld.b.2d.b32.trap {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.trap [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i32 asm "suld.b.3d.b32.zero {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i32 asm "suld.b.3d.b32.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i32 asm "suld.b.3d.b32.trap {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i32 asm "suld.b.a1d.b32.zero {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.zero [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i32 asm "suld.b.a1d.b32.clamp {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.clamp [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i32 asm "suld.b.a1d.b32.trap {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.trap [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i32 asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i32 asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i32 asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surflonglong(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  long long val;

  // CHECK: %0 = tail call i64 asm "suld.b.1d.b64.zero {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.zero [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i64 asm "suld.b.1d.b64.clamp {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.clamp [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i64 asm "suld.b.1d.b64.trap {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.trap [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i64 asm "suld.b.2d.b64.zero {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.zero [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i64 asm "suld.b.2d.b64.clamp {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.clamp [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i64 asm "suld.b.2d.b64.trap {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.trap [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i64 asm "suld.b.3d.b64.zero {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i64 asm "suld.b.3d.b64.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i64 asm "suld.b.3d.b64.trap {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i64 asm "suld.b.a1d.b64.zero {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.zero [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i64 asm "suld.b.a1d.b64.clamp {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.clamp [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i64 asm "suld.b.a1d.b64.trap {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.trap [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surflonglong1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  longlong1 val;

  // CHECK: %0 = tail call i64 asm "suld.b.1d.b64.zero {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.zero [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i64 asm "suld.b.1d.b64.clamp {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.clamp [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i64 asm "suld.b.1d.b64.trap {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.trap [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i64 asm "suld.b.2d.b64.zero {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.zero [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i64 asm "suld.b.2d.b64.clamp {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.clamp [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i64 asm "suld.b.2d.b64.trap {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.trap [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i64 asm "suld.b.3d.b64.zero {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i64 asm "suld.b.3d.b64.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i64 asm "suld.b.3d.b64.trap {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i64 asm "suld.b.a1d.b64.zero {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.zero [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i64 asm "suld.b.a1d.b64.clamp {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.clamp [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i64 asm "suld.b.a1d.b64.trap {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.trap [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfunsignedlonglong(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  unsigned long long val;

  // CHECK: %0 = tail call i64 asm "suld.b.1d.b64.zero {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.zero [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i64 asm "suld.b.1d.b64.clamp {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.clamp [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i64 asm "suld.b.1d.b64.trap {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.trap [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i64 asm "suld.b.2d.b64.zero {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.zero [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i64 asm "suld.b.2d.b64.clamp {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.clamp [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i64 asm "suld.b.2d.b64.trap {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.trap [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i64 asm "suld.b.3d.b64.zero {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i64 asm "suld.b.3d.b64.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i64 asm "suld.b.3d.b64.trap {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i64 asm "suld.b.a1d.b64.zero {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.zero [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i64 asm "suld.b.a1d.b64.clamp {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.clamp [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i64 asm "suld.b.a1d.b64.trap {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.trap [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfulonglong1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  ulonglong1 val;

  // CHECK: %0 = tail call i64 asm "suld.b.1d.b64.zero {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.zero [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call i64 asm "suld.b.1d.b64.clamp {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.clamp [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call i64 asm "suld.b.1d.b64.trap {$0}, [$1, {$2}];", "=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b64.trap [$0, {$1}], {$2};", "l,r,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call i64 asm "suld.b.2d.b64.zero {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.zero [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call i64 asm "suld.b.2d.b64.clamp {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.clamp [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call i64 asm "suld.b.2d.b64.trap {$0}, [$1, {$2, $3}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b64.trap [$0, {$1, $2}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call i64 asm "suld.b.3d.b64.zero {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call i64 asm "suld.b.3d.b64.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call i64 asm "suld.b.3d.b64.trap {$0}, [$1, {$2, $3, $4, $4}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b64.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call i64 asm "suld.b.a1d.b64.zero {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.zero [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call i64 asm "suld.b.a1d.b64.clamp {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.clamp [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call i64 asm "suld.b.a1d.b64.trap {$0}, [$1, {$3, $2}];", "=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b64.trap [$0, {$2, $1}], {$3};", "l,r,r,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call i64 asm "suld.b.a2d.b64.zero {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call i64 asm "suld.b.a2d.b64.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call i64 asm "suld.b.a2d.b64.trap {$0}, [$1, {$4, $2, $3, $3}];", "=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b64.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surffloat(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  float val;

  // CHECK: %0 = tail call contract float asm "suld.b.1d.b32.zero {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.zero [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call contract float asm "suld.b.1d.b32.clamp {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.clamp [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call contract float asm "suld.b.1d.b32.trap {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.trap [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call contract float asm "suld.b.2d.b32.zero {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.zero [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call contract float asm "suld.b.2d.b32.clamp {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.clamp [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call contract float asm "suld.b.2d.b32.trap {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.trap [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call contract float asm "suld.b.3d.b32.zero {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call contract float asm "suld.b.3d.b32.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call contract float asm "suld.b.3d.b32.trap {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call contract float asm "suld.b.a1d.b32.zero {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.zero [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call contract float asm "suld.b.a1d.b32.clamp {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.clamp [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call contract float asm "suld.b.a1d.b32.trap {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.trap [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call contract float asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call contract float asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call contract float asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call contract float asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call contract float asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call contract float asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call contract float asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call contract float asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call contract float asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surffloat1(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  float1 val;

  // CHECK: %0 = tail call contract float asm "suld.b.1d.b32.zero {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.zero [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call contract float asm "suld.b.1d.b32.clamp {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.clamp [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call contract float asm "suld.b.1d.b32.trap {$0}, [$1, {$2}];", "=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.b32.trap [$0, {$1}], {$2};", "l,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call contract float asm "suld.b.2d.b32.zero {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.zero [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call contract float asm "suld.b.2d.b32.clamp {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.clamp [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call contract float asm "suld.b.2d.b32.trap {$0}, [$1, {$2, $3}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.b32.trap [$0, {$1, $2}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call contract float asm "suld.b.3d.b32.zero {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.zero [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call contract float asm "suld.b.3d.b32.clamp {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.clamp [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call contract float asm "suld.b.3d.b32.trap {$0}, [$1, {$2, $3, $4, $4}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.b32.trap [$0, {$1, $2, $3, $3}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call contract float asm "suld.b.a1d.b32.zero {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.zero [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call contract float asm "suld.b.a1d.b32.clamp {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.clamp [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call contract float asm "suld.b.a1d.b32.trap {$0}, [$1, {$3, $2}];", "=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.b32.trap [$0, {$2, $1}], {$3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call contract float asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call contract float asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call contract float asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call contract float asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call contract float asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call contract float asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call contract float asm "suld.b.a2d.b32.zero {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.zero [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call contract float asm "suld.b.a2d.b32.clamp {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.clamp [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call contract float asm "suld.b.a2d.b32.trap {$0}, [$1, {$4, $2, $3, $3}];", "=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.b32.trap [$0, {$3, $1, $2, $2}], {$4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfchar2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  char2 val;

  // CHECK: %0 = tail call { i8, i8 } asm "suld.b.1d.v2.b8.zero {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b8.zero [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i8, i8 } asm "suld.b.1d.v2.b8.clamp {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b8.clamp [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i8, i8 } asm "suld.b.1d.v2.b8.trap {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b8.trap [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i8, i8 } asm "suld.b.2d.v2.b8.zero {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b8.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i8, i8 } asm "suld.b.2d.v2.b8.clamp {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b8.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i8, i8 } asm "suld.b.2d.v2.b8.trap {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b8.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i8, i8 } asm "suld.b.3d.v2.b8.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b8.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i8, i8 } asm "suld.b.3d.v2.b8.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b8.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i8, i8 } asm "suld.b.3d.v2.b8.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b8.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i8, i8 } asm "suld.b.a1d.v2.b8.zero {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b8.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i8, i8 } asm "suld.b.a1d.v2.b8.clamp {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b8.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i8, i8 } asm "suld.b.a1d.v2.b8.trap {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b8.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfuchar2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  uchar2 val;

  // CHECK: %0 = tail call { i8, i8 } asm "suld.b.1d.v2.b8.zero {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b8.zero [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i8, i8 } asm "suld.b.1d.v2.b8.clamp {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b8.clamp [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i8, i8 } asm "suld.b.1d.v2.b8.trap {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b8.trap [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i8, i8 } asm "suld.b.2d.v2.b8.zero {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b8.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i8, i8 } asm "suld.b.2d.v2.b8.clamp {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b8.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i8, i8 } asm "suld.b.2d.v2.b8.trap {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b8.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i8, i8 } asm "suld.b.3d.v2.b8.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b8.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i8, i8 } asm "suld.b.3d.v2.b8.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b8.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i8, i8 } asm "suld.b.3d.v2.b8.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b8.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i8, i8 } asm "suld.b.a1d.v2.b8.zero {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b8.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i8, i8 } asm "suld.b.a1d.v2.b8.clamp {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b8.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i8, i8 } asm "suld.b.a1d.v2.b8.trap {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b8.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i8, i8 } asm "suld.b.a2d.v2.b8.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfshort2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  short2 val;

  // CHECK: %0 = tail call { i16, i16 } asm "suld.b.1d.v2.b16.zero {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b16.zero [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i16, i16 } asm "suld.b.1d.v2.b16.clamp {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b16.clamp [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i16, i16 } asm "suld.b.1d.v2.b16.trap {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b16.trap [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i16, i16 } asm "suld.b.2d.v2.b16.zero {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b16.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i16, i16 } asm "suld.b.2d.v2.b16.clamp {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b16.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i16, i16 } asm "suld.b.2d.v2.b16.trap {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b16.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i16, i16 } asm "suld.b.3d.v2.b16.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b16.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i16, i16 } asm "suld.b.3d.v2.b16.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b16.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i16, i16 } asm "suld.b.3d.v2.b16.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b16.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i16, i16 } asm "suld.b.a1d.v2.b16.zero {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b16.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i16, i16 } asm "suld.b.a1d.v2.b16.clamp {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b16.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i16, i16 } asm "suld.b.a1d.v2.b16.trap {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b16.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}
__device__ void surfushort2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  ushort2 val;

  // CHECK: %0 = tail call { i16, i16 } asm "suld.b.1d.v2.b16.zero {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b16.zero [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i16, i16 } asm "suld.b.1d.v2.b16.clamp {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b16.clamp [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i16, i16 } asm "suld.b.1d.v2.b16.trap {$0, $1}, [$2, {$3}];", "=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b16.trap [$0, {$1}], {$2, $3};", "l,r,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i16, i16 } asm "suld.b.2d.v2.b16.zero {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b16.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i16, i16 } asm "suld.b.2d.v2.b16.clamp {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b16.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i16, i16 } asm "suld.b.2d.v2.b16.trap {$0, $1}, [$2, {$3, $4}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b16.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i16, i16 } asm "suld.b.3d.v2.b16.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b16.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i16, i16 } asm "suld.b.3d.v2.b16.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b16.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i16, i16 } asm "suld.b.3d.v2.b16.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b16.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i16, i16 } asm "suld.b.a1d.v2.b16.zero {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b16.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i16, i16 } asm "suld.b.a1d.v2.b16.clamp {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b16.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i16, i16 } asm "suld.b.a1d.v2.b16.trap {$0, $1}, [$2, {$4, $3}];", "=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b16.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i16, i16 } asm "suld.b.a2d.v2.b16.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfint2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  int2 val;

  // CHECK: %0 = tail call { i32, i32 } asm "suld.b.1d.v2.b32.zero {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.zero [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i32, i32 } asm "suld.b.1d.v2.b32.clamp {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.clamp [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i32, i32 } asm "suld.b.1d.v2.b32.trap {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.trap [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i32, i32 } asm "suld.b.2d.v2.b32.zero {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i32, i32 } asm "suld.b.2d.v2.b32.clamp {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i32, i32 } asm "suld.b.2d.v2.b32.trap {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i32, i32 } asm "suld.b.3d.v2.b32.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i32, i32 } asm "suld.b.3d.v2.b32.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i32, i32 } asm "suld.b.3d.v2.b32.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i32, i32 } asm "suld.b.a1d.v2.b32.zero {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i32, i32 } asm "suld.b.a1d.v2.b32.clamp {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i32, i32 } asm "suld.b.a1d.v2.b32.trap {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfuint2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  uint2 val;

  // CHECK: %0 = tail call { i32, i32 } asm "suld.b.1d.v2.b32.zero {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.zero [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i32, i32 } asm "suld.b.1d.v2.b32.clamp {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.clamp [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i32, i32 } asm "suld.b.1d.v2.b32.trap {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.trap [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i32, i32 } asm "suld.b.2d.v2.b32.zero {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i32, i32 } asm "suld.b.2d.v2.b32.clamp {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i32, i32 } asm "suld.b.2d.v2.b32.trap {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i32, i32 } asm "suld.b.3d.v2.b32.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i32, i32 } asm "suld.b.3d.v2.b32.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i32, i32 } asm "suld.b.3d.v2.b32.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i32, i32 } asm "suld.b.a1d.v2.b32.zero {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i32, i32 } asm "suld.b.a1d.v2.b32.clamp {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i32, i32 } asm "suld.b.a1d.v2.b32.trap {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i32, i32 } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surflonglong2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  longlong2 val;

  // CHECK: %0 = tail call { i64, i64 } asm "suld.b.1d.v2.b64.zero {$0, $1}, [$2, {$3}];", "=l,=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b64.zero [$0, {$1}], {$2, $3};", "l,r,l,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i64, i64 } asm "suld.b.1d.v2.b64.clamp {$0, $1}, [$2, {$3}];", "=l,=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b64.clamp [$0, {$1}], {$2, $3};", "l,r,l,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i64, i64 } asm "suld.b.1d.v2.b64.trap {$0, $1}, [$2, {$3}];", "=l,=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b64.trap [$0, {$1}], {$2, $3};", "l,r,l,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i64, i64 } asm "suld.b.2d.v2.b64.zero {$0, $1}, [$2, {$3, $4}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b64.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i64, i64 } asm "suld.b.2d.v2.b64.clamp {$0, $1}, [$2, {$3, $4}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b64.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i64, i64 } asm "suld.b.2d.v2.b64.trap {$0, $1}, [$2, {$3, $4}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b64.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i64, i64 } asm "suld.b.3d.v2.b64.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b64.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i64, i64 } asm "suld.b.3d.v2.b64.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b64.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i64, i64 } asm "suld.b.3d.v2.b64.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b64.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i64, i64 } asm "suld.b.a1d.v2.b64.zero {$0, $1}, [$2, {$4, $3}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b64.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i64, i64 } asm "suld.b.a1d.v2.b64.clamp {$0, $1}, [$2, {$4, $3}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b64.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i64, i64 } asm "suld.b.a1d.v2.b64.trap {$0, $1}, [$2, {$4, $3}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b64.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfulonglong2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  ulonglong2 val;

  // CHECK: %0 = tail call { i64, i64 } asm "suld.b.1d.v2.b64.zero {$0, $1}, [$2, {$3}];", "=l,=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b64.zero [$0, {$1}], {$2, $3};", "l,r,l,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i64, i64 } asm "suld.b.1d.v2.b64.clamp {$0, $1}, [$2, {$3}];", "=l,=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b64.clamp [$0, {$1}], {$2, $3};", "l,r,l,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i64, i64 } asm "suld.b.1d.v2.b64.trap {$0, $1}, [$2, {$3}];", "=l,=l,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b64.trap [$0, {$1}], {$2, $3};", "l,r,l,l"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i64, i64 } asm "suld.b.2d.v2.b64.zero {$0, $1}, [$2, {$3, $4}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b64.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i64, i64 } asm "suld.b.2d.v2.b64.clamp {$0, $1}, [$2, {$3, $4}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b64.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i64, i64 } asm "suld.b.2d.v2.b64.trap {$0, $1}, [$2, {$3, $4}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b64.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i64, i64 } asm "suld.b.3d.v2.b64.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b64.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i64, i64 } asm "suld.b.3d.v2.b64.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b64.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i64, i64 } asm "suld.b.3d.v2.b64.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b64.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i64, i64 } asm "suld.b.a1d.v2.b64.zero {$0, $1}, [$2, {$4, $3}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b64.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i64, i64 } asm "suld.b.a1d.v2.b64.clamp {$0, $1}, [$2, {$4, $3}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b64.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i64, i64 } asm "suld.b.a1d.v2.b64.trap {$0, $1}, [$2, {$4, $3}];", "=l,=l,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b64.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,l,l"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i64, i64 } asm "suld.b.a2d.v2.b64.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=l,=l,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b64.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,l,l"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surffloat2(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  float2 val;

  // CHECK: %0 = tail call contract { float, float } asm "suld.b.1d.v2.b32.zero {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.zero [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call contract { float, float } asm "suld.b.1d.v2.b32.clamp {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.clamp [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call contract { float, float } asm "suld.b.1d.v2.b32.trap {$0, $1}, [$2, {$3}];", "=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v2.b32.trap [$0, {$1}], {$2, $3};", "l,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call contract { float, float } asm "suld.b.2d.v2.b32.zero {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.zero [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call contract { float, float } asm "suld.b.2d.v2.b32.clamp {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.clamp [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call contract { float, float } asm "suld.b.2d.v2.b32.trap {$0, $1}, [$2, {$3, $4}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v2.b32.trap [$0, {$1, $2}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call contract { float, float } asm "suld.b.3d.v2.b32.zero {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.zero [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call contract { float, float } asm "suld.b.3d.v2.b32.clamp {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.clamp [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call contract { float, float } asm "suld.b.3d.v2.b32.trap {$0, $1}, [$2, {$3, $4, $5, $5}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v2.b32.trap [$0, {$1, $2, $3, $3}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call contract { float, float } asm "suld.b.a1d.v2.b32.zero {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.zero [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call contract { float, float } asm "suld.b.a1d.v2.b32.clamp {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.clamp [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call contract { float, float } asm "suld.b.a1d.v2.b32.trap {$0, $1}, [$2, {$4, $3}];", "=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v2.b32.trap [$0, {$2, $1}], {$3, $4};", "l,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.zero {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.clamp {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call contract { float, float } asm "suld.b.a2d.v2.b32.trap {$0, $1}, [$2, {$5, $3, $4, $4}];", "=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v2.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfchar4(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  char4 val;

  // CHECK: %0 = tail call { i8, i8, i8, i8 } asm "suld.b.1d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b8.zero [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i8, i8, i8, i8 } asm "suld.b.1d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b8.clamp [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i8, i8, i8, i8 } asm "suld.b.1d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b8.trap [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i8, i8, i8, i8 } asm "suld.b.2d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b8.zero [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i8, i8, i8, i8 } asm "suld.b.2d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b8.clamp [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i8, i8, i8, i8 } asm "suld.b.2d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b8.trap [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i8, i8, i8, i8 } asm "suld.b.3d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b8.zero [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i8, i8, i8, i8 } asm "suld.b.3d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b8.clamp [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i8, i8, i8, i8 } asm "suld.b.3d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b8.trap [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i8, i8, i8, i8 } asm "suld.b.a1d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b8.zero [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i8, i8, i8, i8 } asm "suld.b.a1d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b8.clamp [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i8, i8, i8, i8 } asm "suld.b.a1d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b8.trap [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfuchar4(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  uchar4 val;

  // CHECK: %0 = tail call { i8, i8, i8, i8 } asm "suld.b.1d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b8.zero [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i8, i8, i8, i8 } asm "suld.b.1d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b8.clamp [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i8, i8, i8, i8 } asm "suld.b.1d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b8.trap [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i8, i8, i8, i8 } asm "suld.b.2d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b8.zero [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i8, i8, i8, i8 } asm "suld.b.2d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b8.clamp [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i8, i8, i8, i8 } asm "suld.b.2d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b8.trap [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i8, i8, i8, i8 } asm "suld.b.3d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b8.zero [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i8, i8, i8, i8 } asm "suld.b.3d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b8.clamp [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i8, i8, i8, i8 } asm "suld.b.3d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b8.trap [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i8, i8, i8, i8 } asm "suld.b.a1d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b8.zero [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i8, i8, i8, i8 } asm "suld.b.a1d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b8.clamp [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i8, i8, i8, i8 } asm "suld.b.a1d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b8.trap [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i8, i8, i8, i8 } asm "suld.b.a2d.v4.b8.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b8.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfshort4(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  short4 val;

  // CHECK: %0 = tail call { i16, i16, i16, i16 } asm "suld.b.1d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b16.zero [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i16, i16, i16, i16 } asm "suld.b.1d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b16.clamp [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i16, i16, i16, i16 } asm "suld.b.1d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b16.trap [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i16, i16, i16, i16 } asm "suld.b.2d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b16.zero [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i16, i16, i16, i16 } asm "suld.b.2d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b16.clamp [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i16, i16, i16, i16 } asm "suld.b.2d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b16.trap [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i16, i16, i16, i16 } asm "suld.b.3d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b16.zero [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i16, i16, i16, i16 } asm "suld.b.3d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b16.clamp [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i16, i16, i16, i16 } asm "suld.b.3d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b16.trap [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i16, i16, i16, i16 } asm "suld.b.a1d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b16.zero [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i16, i16, i16, i16 } asm "suld.b.a1d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b16.clamp [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i16, i16, i16, i16 } asm "suld.b.a1d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b16.trap [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfushort4(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  ushort4 val;

  // CHECK: %0 = tail call { i16, i16, i16, i16 } asm "suld.b.1d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b16.zero [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i16, i16, i16, i16 } asm "suld.b.1d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b16.clamp [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i16, i16, i16, i16 } asm "suld.b.1d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$5}];", "=h,=h,=h,=h,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b16.trap [$0, {$1}], {$2, $3, $4, $5};", "l,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i16, i16, i16, i16 } asm "suld.b.2d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b16.zero [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i16, i16, i16, i16 } asm "suld.b.2d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b16.clamp [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i16, i16, i16, i16 } asm "suld.b.2d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$5, $6}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b16.trap [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i16, i16, i16, i16 } asm "suld.b.3d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b16.zero [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i16, i16, i16, i16 } asm "suld.b.3d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b16.clamp [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i16, i16, i16, i16 } asm "suld.b.3d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b16.trap [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i16, i16, i16, i16 } asm "suld.b.a1d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b16.zero [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i16, i16, i16, i16 } asm "suld.b.a1d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b16.clamp [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i16, i16, i16, i16 } asm "suld.b.a1d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$6, $5}];", "=h,=h,=h,=h,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b16.trap [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i16, i16, i16, i16 } asm "suld.b.a2d.v4.b16.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=h,=h,=h,=h,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b16.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,h,h,h,h"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfint4(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  int4 val;

  // CHECK: %0 = tail call { i32, i32, i32, i32 } asm "suld.b.1d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.zero [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i32, i32, i32, i32 } asm "suld.b.1d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.clamp [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i32, i32, i32, i32 } asm "suld.b.1d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.trap [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i32, i32, i32, i32 } asm "suld.b.2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.zero [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i32, i32, i32, i32 } asm "suld.b.2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.clamp [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i32, i32, i32, i32 } asm "suld.b.2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.trap [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i32, i32, i32, i32 } asm "suld.b.3d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.zero [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i32, i32, i32, i32 } asm "suld.b.3d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.clamp [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i32, i32, i32, i32 } asm "suld.b.3d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.trap [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i32, i32, i32, i32 } asm "suld.b.a1d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.zero [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i32, i32, i32, i32 } asm "suld.b.a1d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.clamp [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i32, i32, i32, i32 } asm "suld.b.a1d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.trap [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surfuint4(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  uint4 val;

  // CHECK: %0 = tail call { i32, i32, i32, i32 } asm "suld.b.1d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.zero [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call { i32, i32, i32, i32 } asm "suld.b.1d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.clamp [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call { i32, i32, i32, i32 } asm "suld.b.1d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.trap [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call { i32, i32, i32, i32 } asm "suld.b.2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.zero [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call { i32, i32, i32, i32 } asm "suld.b.2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.clamp [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call { i32, i32, i32, i32 } asm "suld.b.2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.trap [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call { i32, i32, i32, i32 } asm "suld.b.3d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.zero [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call { i32, i32, i32, i32 } asm "suld.b.3d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.clamp [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call { i32, i32, i32, i32 } asm "suld.b.3d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.trap [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call { i32, i32, i32, i32 } asm "suld.b.a1d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.zero [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call { i32, i32, i32, i32 } asm "suld.b.a1d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.clamp [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call { i32, i32, i32, i32 } asm "suld.b.a1d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.trap [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call { i32, i32, i32, i32 } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}

__device__ void surffloat4(cudaSurfaceObject_t surf, int x, int y, int z, int layer, int face, int layerface) {
  float4 val;

  // CHECK: %0 = tail call contract { float, float, float, float } asm "suld.b.1d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.zero [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeZero);
  // CHECK: %1 = tail call contract { float, float, float, float } asm "suld.b.1d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.clamp [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeClamp);
  // CHECK: %2 = tail call contract { float, float, float, float } asm "suld.b.1d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5}];", "=r,=r,=r,=r,l,r"
  __nv_tex_surf_handler("__isurf1Dread", &val, surf, x, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.1d.v4.b32.trap [$0, {$1}], {$2, $3, $4, $5};", "l,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, surf, x, cudaBoundaryModeTrap);

  // CHECK: %3 = tail call contract { float, float, float, float } asm "suld.b.2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.zero [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeZero);
  // CHECK: %4 = tail call contract { float, float, float, float } asm "suld.b.2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.clamp [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeClamp);
  // CHECK: %5 = tail call contract { float, float, float, float } asm "suld.b.2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5, $6}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf2Dread", &val, surf, x, y, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.2d.v4.b32.trap [$0, {$1, $2}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, surf, x, y, cudaBoundaryModeTrap);

  // CHECK: %6 = tail call contract { float, float, float, float } asm "suld.b.3d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.zero [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeZero);
  // CHECK: %7 = tail call contract { float, float, float, float } asm "suld.b.3d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.clamp [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeClamp);
  // CHECK: %8 = tail call contract { float, float, float, float } asm "suld.b.3d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$5, $6, $7, $7}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf3Dread", &val, surf, x, y, z, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.3d.v4.b32.trap [$0, {$1, $2, $3, $3}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, surf, x, y, z, cudaBoundaryModeTrap);

  // CHECK: %9 = tail call contract { float, float, float, float } asm "suld.b.a1d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.zero [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeZero);
  // CHECK: %10 = tail call contract { float, float, float, float } asm "suld.b.a1d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.clamp [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeClamp);
  // CHECK: %11 = tail call contract { float, float, float, float } asm "suld.b.a1d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$6, $5}];", "=r,=r,=r,=r,l,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredread", &val, surf, x, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a1d.v4.b32.trap [$0, {$2, $1}], {$3, $4, $5, $6};", "l,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, surf, x, layer, cudaBoundaryModeTrap);

  // CHECK: %12 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeZero);
  // CHECK: %13 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeClamp);
  // CHECK: %14 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredread", &val, surf, x, y, layer, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, surf, x, y, layer, cudaBoundaryModeTrap);

  // CHECK: %15 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeZero);
  // CHECK: %16 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeClamp);
  // CHECK: %17 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapread", &val, surf, x, y, face, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, surf, x, y, face, cudaBoundaryModeTrap);

  // CHECK: %18 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.zero {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.zero [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeZero);
  // CHECK: %19 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.clamp {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.clamp [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeClamp);
  // CHECK: %20 = tail call contract { float, float, float, float } asm "suld.b.a2d.v4.b32.trap {$0, $1, $2, $3}, [$4, {$7, $5, $6, $6}];", "=r,=r,=r,=r,l,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredread", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
  // CHECK: tail call void asm sideeffect "sust.b.a2d.v4.b32.trap [$0, {$3, $1, $2, $2}], {$4, $5, $6, $7};", "l,r,r,r,r,r,r,r"
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, surf, x, y, layerface, cudaBoundaryModeTrap);
}
