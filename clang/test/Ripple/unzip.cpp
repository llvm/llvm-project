// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S %s -O2 -fenable-ripple -emit-llvm -o - | FileCheck %s

// Tests for the zip.h header-only lib

#include <ripple.h>
#include <ripple/zip.h>

using namespace rzip;

extern "C" {

// Simple unzipping of pairs
// CHECK-LABEL: define dso_local void @unzip_pair0
void unzip_pair0(float * dst, float *src) {
    constexpr size_t nv0 = 32;
    constexpr size_t nv1 = 2;
    ripple_block_t BS = ripple_set_block_shape(0, nv0, nv1);
    size_t v0 = ripple_id(BS, 0);
    size_t v1 = ripple_id(BS, 1);
    size_t v = nv0*v1 + v0;
    float x = src[v];
    float unzipped = ripple_shuffle(x, shuffle_unzip<2, 0, 0>);
    // CHECK: %{{.*}} = shufflevector <64 x float> %{{.*}}, <64 x float> poison, <64 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63>
    dst[v] = unzipped;
}

// Unzipping of pairs involving ripple_slice
// CHECK-LABEL: define dso_local void @unzip_pair1
void unzip_pair1(float * dst, float *src) {
    constexpr size_t nv0 = 32;
    constexpr size_t nv1 = 2;
    ripple_block_t BS = ripple_set_block_shape(0, nv0, nv1);
    size_t v0 = ripple_id(BS, 0);
    size_t v1 = ripple_id(BS, 1);
    float x = src[nv0*v1 + v0];
    float unzipped = ripple_shuffle(x, shuffle_unzip<2, 0, 0>);
    // CHECK: %{{.*}} = shufflevector <64 x float> %{{.*}}, <64 x float> poison, <{{32|64}} x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62
    float even = ripple_slice(unzipped, -1, 0);
    float odd = ripple_slice(unzipped, -1, 1);
    dst[v0] = even;
    dst[v0 + nv1*nv0 ] = odd;
}

// Using the presumably less-efficient 1-d library method
// CHECK-LABEL: define dso_local void @unzip_pair2
void unzip_pair2(float * dst, float * src) {
    constexpr size_t nv0 = 32;
    ripple_block_t BS = ripple_set_block_shape(0, nv0);
    size_t v0 = ripple_id(BS, 0);
    rzip_unzip_pairs(v0, nv0, src[v0], src[nv0 + v0], dst[v0], dst[nv0 + v0]);
}


// CHECK-LABEL: define dso_local void @unzip_triple0
void unzip_triple0(float *dst, float *src) {
    constexpr size_t nv0 = 32;
    constexpr size_t nv1 = 3;
    ripple_block_t BS = ripple_set_block_shape(0, nv0, nv1);
    size_t v0 = ripple_id(BS, 0);
    size_t v1 = ripple_id(BS, 1);
    size_t v = nv0*v1 + v0;
    dst[v] = ripple_shuffle(src[v], shuffle_unzip<3, 0, 0>);
    // CHECK: %{{.*}} = shufflevector <96 x float> %{{.*}}, <96 x float> poison, <96 x i32> <i32 0, i32 3, i32 6, i32 9, i32 12, i32 15, i32 18, i32 21, i32 24, i32 27, i32 30, i32 33, i32 36, i32 39, i32 42, i32 45, i32 48, i32 51, i32 54, i32 57, i32 60, i32 63, i32 66, i32 69, i32 72, i32 75, i32 78, i32 81, i32 84, i32 87, i32 90, i32 93, i32 1, i32 4, i32 7, i32 10, i32 13, i32 16, i32 19, i32 22, i32 25, i32 28, i32 31, i32 34, i32 37, i32 40, i32 43, i32 46, i32 49, i32 52, i32 55, i32 58, i32 61, i32 64, i32 67, i32 70, i32 73, i32 76, i32 79, i32 82, i32 85, i32 88, i32 91, i32 94, i32 2, i32 5, i32 8, i32 11, i32 14, i32 17, i32 20, i32 23, i32 26, i32 29, i32 32, i32 35, i32 38, i32 41, i32 44, i32 47, i32 50, i32 53, i32 56, i32 59, i32 62, i32 65, i32 68, i32 71, i32 74, i32 77, i32 80, i32 83, i32 86, i32 89, i32 92, i32 95>
}

// CHECK-LABEL: define dso_local void @unzip_triple1
void unzip_triple1(float *dst, float *src) {
    constexpr size_t nv0 = 32;
    constexpr size_t nv1 = 3;
    ripple_block_t BS = ripple_set_block_shape(0, nv0, nv1);
    size_t v0 = ripple_id(BS, 0);
    size_t v1 = ripple_id(BS, 1);
    size_t v = nv0*v1 + v0;
    float x = src[v];
    float unzipped = ripple_shuffle(x, shuffle_unzip<3, 0, 0>);
    float y0 = ripple_slice(unzipped, -1, 0);
    float y1 = ripple_slice(unzipped, -1, 1);
    float y2 = ripple_slice(unzipped, -1, 2);
    dst[v0] = y0;
    dst[v0 + nv0] = y1;
    dst[v0 + 2*nv0] = y2;
}

// Using the presumably less-efficient 1-d library method
// CHECK-LABEL: define dso_local void @unzip_triple2
void unzip_triple2(float * dst, float * src) {
    constexpr size_t nv0 = 32;
    ripple_block_t BS = ripple_set_block_shape(0, nv0);
    size_t v0 = ripple_id(BS, 0);
    rzip_unzip_triples(v0, nv0, src[v0], src[nv0 + v0], src[2*nv0 + v0],
        dst[v0], dst[nv0 + v0], dst[2*nv0 + v0]);
}

// CHECK-LABEL: define dso_local void @unzip_quad0
void unzip_quad0(float *dst, float *src) {
    constexpr size_t nv0 = 32;
    constexpr size_t nv1 = 4;
    ripple_block_t BS = ripple_set_block_shape(0, nv0, nv1);
    size_t v0 = ripple_id(BS, 0);
    size_t v1 = ripple_id(BS, 1);
    size_t v = nv0*v1 + v0;
    dst[v] = ripple_shuffle(src[v], shuffle_unzip<4, 0, 0>);
    // CHECK: %{{.*}} = shufflevector <128 x float> %{{.*}}, <128 x float> poison, <128 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28, i32 32, i32 36, i32 40, i32 44, i32 48, i32 52, i32 56, i32 60, i32 64, i32 68, i32 72, i32 76, i32 80, i32 84, i32 88, i32 92, i32 96, i32 100, i32 104, i32 108, i32 112, i32 116, i32 120, i32 124, i32 1, i32 5, i32 9, i32 13, i32 17, i32 21, i32 25, i32 29, i32 33, i32 37, i32 41, i32 45, i32 49, i32 53, i32 57, i32 61, i32 65, i32 69, i32 73, i32 77, i32 81, i32 85, i32 89, i32 93, i32 97, i32 101, i32 105, i32 109, i32 113, i32 117, i32 121, i32 125, i32 2, i32 6, i32 10, i32 14, i32 18, i32 22, i32 26, i32 30, i32 34, i32 38, i32 42, i32 46, i32 50, i32 54, i32 58, i32 62, i32 66, i32 70, i32 74, i32 78, i32 82, i32 86, i32 90, i32 94, i32 98, i32 102, i32 106, i32 110, i32 114, i32 118, i32 122, i32 126, i32 3, i32 7, i32 11, i32 15, i32 19, i32 23, i32 27, i32 31, i32 35, i32 39, i32 43, i32 47, i32 51, i32 55, i32 59, i32 63, i32 67, i32 71, i32 75, i32 79, i32 83, i32 87, i32 91, i32 95, i32 99, i32 103, i32 107, i32 111, i32 115, i32 119, i32 123, i32 127>
}

// CHECK-LABEL: define dso_local void @unzip_quad1
void unzip_quad1(float *dst, float *src) {
    constexpr size_t nv0 = 32;
    constexpr size_t nv1 = 4;
    ripple_block_t BS = ripple_set_block_shape(0, nv0, nv1);
    size_t v0 = ripple_id(BS, 0);
    size_t v1 = ripple_id(BS, 1);
    size_t v = nv0*v1 + v0;
    size_t nv = nv0 * nv1;
    float x = src[v];
    float unzipped = ripple_shuffle(x, shuffle_unzip<4, 0, 0>);
    float y0 = ripple_slice(unzipped, -1, 0);
    float y1 = ripple_slice(unzipped, -1, 1);
    float y2 = ripple_slice(unzipped, -1, 2);
    float y3 = ripple_slice(unzipped, -1, 3);
    dst[v0] = y0;
    dst[v0 + nv0] = y1;
    dst[v0 + 2*nv0] = y2;
    dst[v0 + 3*nv0] = y3;
}

// Using the presumably less-efficient 1-d library method
// CHECK-LABEL: define dso_local void @unzip_quad2
void unzip_quad2(float * dst, float * src) {
    constexpr size_t nv0 = 32;
    ripple_block_t BS = ripple_set_block_shape(0, nv0);
    size_t v0 = ripple_id(BS, 0);
    rzip_unzip_quads(v0, nv0, src[v0], src[nv0 + v0], src[2*nv0 + v0], src[3*nv0 + v0],
                dst[v0], dst[nv0 + v0], dst[2*nv0 + v0], dst[3*nv0 + v0]);
}

}