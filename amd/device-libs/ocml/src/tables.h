/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// Table stuff

#undef USE_TABLESTRUCT

#ifdef USE_TABLESTRUCT

struct __tbl_mem_s {
    float M32_J0[72];
    float M32_J1[72];
    float M32_Y0[162]
    float M32_Y1[162]
    double M64_J0[120];
    double M64_J1[120];
    double M64_Y0[270];
    double M64_Y1[270];
};

extern __constant struct __tbl_mem_s __tbl_mem;

#define USE_TABLE(TYPE,PTR,NAME) \
    __constant TYPE * PTR = __ocmltbl_mem . NAME

#else

#define TABLE_MANGLE(NAME) __ocmltbl_##NAME

extern __constant float TABLE_MANGLE(M32_J0)[];
extern __constant float TABLE_MANGLE(M32_J1)[];
extern __constant float TABLE_MANGLE(M32_Y0)[];
extern __constant float TABLE_MANGLE(M32_Y1)[];
extern __constant double TABLE_MANGLE(M64_J0)[];
extern __constant double TABLE_MANGLE(M64_J1)[];
extern __constant double TABLE_MANGLE(M64_Y0)[];
extern __constant double TABLE_MANGLE(M64_Y1)[];

#define USE_TABLE(TYPE,PTR,NAME) \
    __constant TYPE * PTR = TABLE_MANGLE(NAME)

#endif

