#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Set compiler flags and definitions
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_L1")
get_property(FLAGS GLOBAL PROPERTY "FLAGS_L1")

set(FLAGS "${FLAGS} -Wno-attributes")
set(DEFINITIONS "${DEFINITIONS} -DPGI")
set(SRC_VECTOR fs_sincos_vector.cpp)
set(SRC_SCALAR fs_sincos_scalar.cpp)

set(AVX2_FLAGS "${FLAGS} -mtune=core-avx2 -march=core-avx2")
set(AVX512_FLAGS "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512")
set(KNL_FLAGS "${FLAGS} -mtune=knl -march=knl")

# SINCOS
libmath_add_object_library("${SRC_SCALAR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DSINCOS" "fs_sincos_1_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DSINCOS -DVL=4" "fs_sincos_4_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DSINCOS -DVL=8" "fs_sincos_8_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX512_FLAGS}" "${DEFINITIONS} -DSINCOS -DVL=16" "fs_sincos_16_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${KNL_FLAGS}" "${DEFINITIONS} -DSINCOS -DVL=16" "fs_sincos_16_knl")

# COSINE
libmath_add_object_library("${SRC_SCALAR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DCOSINE" "fs_cos_1_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DCOSINE -DVL=4" "fs_cos_4_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DCOSINE -DVL=8" "fs_cos_8_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX512_FLAGS}" "${DEFINITIONS} -DCOSINE -DVL=16" "fs_cos_16_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${KNL_FLAGS}" "${DEFINITIONS} -DCOSINE -DVL=16" "fs_cos_16_knl")

# SINE
libmath_add_object_library("${SRC_SCALAR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DSINE" "fs_sin_1_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DSINE -DVL=4" "fs_sin_4_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX2_FLAGS}" "${DEFINITIONS} -DSINE -DVL=8" "fs_sin_8_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${AVX512_FLAGS}" "${DEFINITIONS} -DSINE -DVL=16" "fs_sin_16_avx2")
libmath_add_object_library("${SRC_VECTOR}" "${KNL_FLAGS}" "${DEFINITIONS} -DSINE -DVL=16" "fs_sin_16_knl")
