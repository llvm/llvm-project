
// Table stuff

#undef USE_TABLESTRUCT

#ifdef USE_TABLESTRUCT

struct __tbl_mem_s {
#if 0
    double2 M64_ATAN_JBY256[241];
#endif

#ifdef TABLE_BASED_ATAN2
    float M32_ATAN2_JBY256[241];
#endif

    double M64_CBRT_INV[257];
    double2 M64_CBRT[257];
    double2 M64_CBRT_REM[5];
    float2 M32_CBRT[129];
    double2 M64_EXP_EP[64];
    float M32_EXP[65];
    float2 M32_EXP_EP[65];
    double2 M64_LOGE_EP[65];
    float2 M32_LOG2[129];
    float2 M32_LOG10[129];
    float2 M32_LOGE[129];
    float M32_LOG_INV[129];
    float2 M32_LOG_INV_EP[129];
    uint M64_PIBITS[37];
    double2 M64_POWLOG[258];
    double2 M64_LOG_F_INV[258];
    double2 M64_SINH[37];
    double2 M64_COSH[37];
    float2 M32_SINHCOSH[37];
    float M32_RSQRT[64];
    double M64_RSQRT[128];
    float M32_ERF[55];
    double M64_ERF[117];
    float2 M32_RCBRT[129];
    double2 M64_RCBRT[257];
    double2 M64_RCBRT_REM[5];
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

#if 0
extern __constant double2 TABLE_MANGLE(M64_ATAN_JBY256)[];
#endif

#ifdef TABLE_BASED_ATAN2
extern __constant float TABLE_MANGLE(M32_ATAN2_JBY256)[];
#endif

extern __constant double TABLE_MANGLE(M64_CBRT_INV)[];
extern __constant double2 TABLE_MANGLE(M64_CBRT)[];
extern __constant double2 TABLE_MANGLE(M64_CBRT_REM)[];
extern __constant float2 TABLE_MANGLE(M32_CBRT)[];
extern __constant double TABLE_MANGLE(M64_ERF)[];
extern __constant float TABLE_MANGLE(M32_ERF)[];
extern __constant double2 TABLE_MANGLE(M64_EXP_EP)[];
extern __constant float TABLE_MANGLE(M32_EXP)[];
extern __constant float2 TABLE_MANGLE(M32_EXP_EP)[];
extern __constant double2 TABLE_MANGLE(M64_LOGE_EP)[];
extern __constant float2 TABLE_MANGLE(M32_LOG2)[];
extern __constant float2 TABLE_MANGLE(M32_LOG10)[];
extern __constant float2 TABLE_MANGLE(M32_LOGE)[];
extern __constant float TABLE_MANGLE(M32_LOG_INV)[];
extern __constant float2 TABLE_MANGLE(M32_LOG_INV_EP)[];
extern __constant uint TABLE_MANGLE(M64_PIBITS)[];
extern __constant double2 TABLE_MANGLE(M64_POWLOG)[];
extern __constant double2 TABLE_MANGLE(M64_LOG_F_INV)[];
extern __constant double2 TABLE_MANGLE(M64_RCBRT)[];
extern __constant double2 TABLE_MANGLE(M64_RCBRT_REM)[];
extern __constant float2 TABLE_MANGLE(M32_RCBRT)[];
extern __constant double TABLE_MANGLE(M64_RSQRT)[];
extern __constant float TABLE_MANGLE(M32_RSQRT)[];
extern __constant double2 TABLE_MANGLE(M64_SINH)[];
extern __constant double2 TABLE_MANGLE(M64_COSH)[];
extern __constant float2 TABLE_MANGLE(M32_SINHCOSH)[];
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

