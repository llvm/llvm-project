
#ifndef OCKL_H
#define OCKL_H

// This C header declares the functions provided by the OCKL library
// Aspects of this library's behavior can be controlled via the 
// oclc library.  See the oclc header for further information

#define _MANGLE3x(P,N,S) P##_##N##S
#define MANGLE3x(P,N,S) _MANGLE3x(P,N,S)
#define _MANGLE3(P,N,S) P##_##N##_##S
#define MANGLE3(P,N,S) _MANGLE3(P,N,S)
#define OCKL_MANGLE_T(N,T) MANGLE3(__ockl, N, T)
#define OCKL_MANGLE_Tx(N,T) MANGLE3x(__ockl, N, T)
#define OCKL_MANGLE_I32(N) OCKL_MANGLE_T(N, i32)
#define OCKL_MANGLE_U32(N) OCKL_MANGLE_T(N, u32)
#define OCKL_MANGLE_I64(N) OCKL_MANGLE_T(N, i64)
#define OCKL_MANGLE_U64(N) OCKL_MANGLE_T(N, u64)

#define DECL_OCKL_NULLARY_U32(N) extern uint OCKL_MANGLE_U32(N)(void);
#define _DECL_X_OCKL_NULLARY_U32(A,N) extern __attribute__((A)) uint OCKL_MANGLE_U32(N)(void);
#define DECL_PURE_OCKL_NULLARY_U32(N) _DECL_X_OCKL_NULLARY_U32(pure, N)
#define DECL_CONST_OCKL_NULLARY_U32(N) _DECL_X_OCKL_NULLARY_U32(const, N)

#define DECL_OCKL_UNARY_I32(N) extern int OCKL_MANGLE_I32(N)(int);
#define _DECL_X_OCKL_UNARY_I32(A,N) extern __attribute__((A)) int OCKL_MANGLE_I32(N)(int);
#define DECL_PURE_OCKL_UNARY_I32(N) _DECL_X_OCKL_UNARY_I32(pure, N)
#define DECL_CONST_OCKL_UNARY_I32(N) _DECL_X_OCKL_UNARY_I32(const, N)

#define DECL_OCKL_UNARY_I64(N) extern long OCKL_MANGLE_I64(N)(long);
#define _DECL_X_OCKL_UNARY_I64(A,N) extern __attribute__((A)) long OCKL_MANGLE_I64(N)(long);
#define DECL_PURE_OCKL_UNARY_I64(N) _DECL_X_OCKL_UNARY_I64(pure, N)
#define DECL_CONST_OCKL_UNARY_I64(N) _DECL_X_OCKL_UNARY_I64(const, N)

#define DECL_OCKL_UNARY_U32(N) extern uint OCKL_MANGLE_U32(N)(uint);
#define _DECL_X_OCKL_UNARY_U32(A,N) extern __attribute__((A)) uint OCKL_MANGLE_U32(N)(uint);
#define DECL_PURE_OCKL_UNARY_U32(N) _DECL_X_OCKL_UNARY_U32(pure, N)
#define DECL_CONST_OCKL_UNARY_U32(N) _DECL_X_OCKL_UNARY_U32(const, N)

#define DECL_OCKL_UNARY_U64(N) extern ulong OCKL_MANGLE_U64(N)(ulong);
#define _DECL_X_OCKL_UNARY_U64(A,N) extern __attribute__((A)) ulong OCKL_MANGLE_U64(N)(ulong);
#define DECL_PURE_OCKL_UNARY_U64(N) _DECL_X_OCKL_UNARY_U64(pure, N)
#define DECL_CONST_OCKL_UNARY_U64(N) _DECL_X_OCKL_UNARY_U64(const, N)

#define DECL_OCKL_BINARY_I32(N) extern int OCKL_MANGLE_I32(N)(int,int);
#define _DECL_X_OCKL_BINARY_I32(A,N) extern __attribute__((A)) int OCKL_MANGLE_I32(N)(int,int);
#define DECL_PURE_OCKL_BINARY_I32(N) _DECL_X_OCKL_BINARY_I32(pure, N)
#define DECL_CONST_OCKL_BINARY_I32(N) _DECL_X_OCKL_BINARY_I32(const, N)

#define DECL_OCKL_BINARY_I64(N) extern long OCKL_MANGLE_I64(N)(long,long);
#define _DECL_X_OCKL_BINARY_I64(A,N) extern __attribute__((A)) long OCKL_MANGLE_I64(N)(long,long);
#define DECL_PURE_OCKL_BINARY_I64(N) _DECL_X_OCKL_BINARY_I64(pure, N)
#define DECL_CONST_OCKL_BINARY_I64(N) _DECL_X_OCKL_BINARY_I64(const, N)

#define DECL_OCKL_BINARY_U32(N) extern uint OCKL_MANGLE_U32(N)(uint,uint);
#define _DECL_X_OCKL_BINARY_U32(A,N) extern __attribute__((A)) uint OCKL_MANGLE_U32(N)(uint,uint);
#define DECL_PURE_OCKL_BINARY_U32(N) _DECL_X_OCKL_BINARY_U32(pure, N)
#define DECL_CONST_OCKL_BINARY_U32(N) _DECL_X_OCKL_BINARY_U32(const, N)

#define DECL_OCKL_BINARY_U64(N) extern ulong OCKL_MANGLE_U64(N)(ulong,ulong);
#define _DECL_X_OCKL_BINARY_U64(A,N) extern __attribute__((A)) ulong OCKL_MANGLE_U64(N)(ulong,ulong);
#define DECL_PURE_OCKL_BINARY_U64(N) _DECL_X_OCKL_BINARY_U64(pure, N)
#define DECL_CONST_OCKL_BINARY_U64(N) _DECL_X_OCKL_BINARY_U64(const, N)

DECL_CONST_OCKL_UNARY_U32(clz)
DECL_CONST_OCKL_UNARY_U32(ctz)
DECL_CONST_OCKL_UNARY_U32(popcount)

DECL_CONST_OCKL_BINARY_I32(add_sat)
DECL_CONST_OCKL_BINARY_U32(add_sat)
DECL_CONST_OCKL_BINARY_I64(add_sat)
DECL_CONST_OCKL_BINARY_U64(add_sat)

DECL_CONST_OCKL_BINARY_I32(sub_sat)
DECL_CONST_OCKL_BINARY_U32(sub_sat)
DECL_CONST_OCKL_BINARY_I64(sub_sat)
DECL_CONST_OCKL_BINARY_U64(sub_sat)

DECL_CONST_OCKL_BINARY_I32(mul_hi)
DECL_CONST_OCKL_BINARY_U32(mul_hi)
DECL_CONST_OCKL_BINARY_I64(mul_hi)
DECL_CONST_OCKL_BINARY_U64(mul_hi)

DECL_CONST_OCKL_BINARY_I32(mul24)
DECL_CONST_OCKL_BINARY_U32(mul24)

DECL_OCKL_NULLARY_U32(activelane)

#define SSHARP __constant uint *
#define TSHARP __constant uint *

extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_load,1D)(TSHARP i, int c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_load,1Da)(TSHARP i, int2 c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_load,1Db)(TSHARP i, int c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_load,2D)(TSHARP i, int2 c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_load,2Da)(TSHARP i, int4 c);
extern __attribute__((pure)) float OCKL_MANGLE_T(image_load,2Dad)(TSHARP i, int4 c);
extern __attribute__((pure)) float OCKL_MANGLE_T(image_load,2Dd)(TSHARP i, int2 c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_load,3D)(TSHARP i, int4 c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_loadh,1D)(TSHARP i, int c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_loadh,1Da)(TSHARP i, int2 c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_loadh,1Db)(TSHARP i, int c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_loadh,2D)(TSHARP i, int2 c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_loadh,2Da)(TSHARP i, int4 c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_loadh,3D)(TSHARP i, int4 c);

extern void OCKL_MANGLE_T(image_store,1D)(TSHARP i, int c, float4 p);
extern void OCKL_MANGLE_T(image_store,1Da)(TSHARP i, int2 c, float4 p);
extern void OCKL_MANGLE_T(image_store,1Db)(TSHARP i, int c, float4 p);
extern void OCKL_MANGLE_T(image_store,2D)(TSHARP i, int2 c, float4 p);
extern void OCKL_MANGLE_T(image_store,2Da)(TSHARP i, int4 c, float4 p);
extern void OCKL_MANGLE_T(image_store,2Dad)(TSHARP i, int4 c, float4 p);
extern void OCKL_MANGLE_T(image_store,2Dd)(TSHARP i, int2 c, float4 p);
extern void OCKL_MANGLE_T(image_store,3D)(TSHARP i, int4 c, float4 p);
extern void OCKL_MANGLE_T(image_store_lod,1D)(TSHARP i, int c, int l, float4 p);
extern void OCKL_MANGLE_T(image_store_lod,1Da)(TSHARP i, int2 c, int l, float4 p);
extern void OCKL_MANGLE_T(image_store_lod,2D)(TSHARP i, int2 c, int l, float4 p);
extern void OCKL_MANGLE_T(image_store_lod,2Da)(TSHARP i, int4 c, int l, float4 p);
extern void OCKL_MANGLE_T(image_store_lod,2Dad)(TSHARP i, int4 c, int l, float4 p);
extern void OCKL_MANGLE_T(image_store_lod,2Dd)(TSHARP i, int2 c, int l, float4 p);
extern void OCKL_MANGLE_T(image_store_lod,3D)(TSHARP i, int4 c, int l, float4 p);
extern void OCKL_MANGLE_T(image_storeh,1D)(TSHARP i, int c, half4 p);
extern void OCKL_MANGLE_T(image_storeh,1Da)(TSHARP i, int2 c, half4 p);
extern void OCKL_MANGLE_T(image_storeh,1Db)(TSHARP i, int c, half4 p);
extern void OCKL_MANGLE_T(image_storeh,2D)(TSHARP i, int2 c, half4 p);
extern void OCKL_MANGLE_T(image_storeh,2Da)(TSHARP i, int4 c, half4 p);
extern void OCKL_MANGLE_T(image_storeh,3D)(TSHARP i, int4 c, half4 p);
extern void OCKL_MANGLE_T(image_storeh_lod,1D)(TSHARP i, int c, int l, half4 p);
extern void OCKL_MANGLE_T(image_storeh_lod,1Da)(TSHARP i, int2 c, int l, half4 p);
extern void OCKL_MANGLE_T(image_storeh_lod,2D)(TSHARP i, int2 c, int l, half4 p);
extern void OCKL_MANGLE_T(image_storeh_lod,2Da)(TSHARP i, int4 c, int l, half4 p);
extern void OCKL_MANGLE_T(image_storeh_lod,3D)(TSHARP i, int4 c, int l, half4 p);

extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample,1D)(TSHARP i, SSHARP s, float c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample,1Da)(TSHARP i, SSHARP s, float2 c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample,2D)(TSHARP i, SSHARP s, float2 c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample,2Da)(TSHARP i, SSHARP s, float4 c);
extern __attribute__((pure)) float OCKL_MANGLE_T(image_sample,2Dad)(TSHARP i, SSHARP s, float4 c);
extern __attribute__((pure)) float OCKL_MANGLE_T(image_sample,2Dd)(TSHARP i, SSHARP s, float2 c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample,3D)(TSHARP i, SSHARP s, float4 c);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_grad,1D)(TSHARP i, SSHARP s, float c, float dx, float dy);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_grad,1Da)(TSHARP i, SSHARP s, float2 c, float dx, float dy);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_grad,2D)(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_grad,2Da)(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy);
extern __attribute__((pure)) float OCKL_MANGLE_T(image_sample_grad,2Dad)(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy);
extern __attribute__((pure)) float OCKL_MANGLE_T(image_sample_grad,2Dd)(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_grad,3D)(TSHARP i, SSHARP s, float4 c, float4 dx, float4 dy);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_lod,1D)(TSHARP i, SSHARP s, float c, float l);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_lod,1Da)(TSHARP i, SSHARP s, float2 c, float l);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_lod,2D)(TSHARP i, SSHARP s, float2 c, float l);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_lod,2Da)(TSHARP i, SSHARP s, float4 c, float l);
extern __attribute__((pure)) float OCKL_MANGLE_T(image_sample_lod,2Dad)(TSHARP i, SSHARP s, float4 c, float l);
extern __attribute__((pure)) float OCKL_MANGLE_T(image_sample_lod,2Dd)(TSHARP i, SSHARP s, float2 c, float l);
extern __attribute__((pure)) float4 OCKL_MANGLE_T(image_sample_lod,3D)(TSHARP i, SSHARP s, float4 c, float l);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh,1D)(TSHARP i, SSHARP s, float c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh,1Da)(TSHARP i, SSHARP s, float2 c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh,2D)(TSHARP i, SSHARP s, float2 c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh,2Da)(TSHARP i, SSHARP s, float4 c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh,3D)(TSHARP i, SSHARP s, float4 c);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_grad,1D)(TSHARP i, SSHARP s, float c, float dx, float dy);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_grad,1Da)(TSHARP i, SSHARP s, float2 c, float dx, float dy);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_grad,2D)(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_grad,2Da)(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_grad,3D)(TSHARP i, SSHARP s, float4 c, float4 dx, float4 dy);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_lod,1D)(TSHARP i, SSHARP s, float c, float l);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_lod,1Da)(TSHARP i, SSHARP s, float2 c, float l);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_lod,2D)(TSHARP i, SSHARP s, float2 c, float l);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_lod,2Da)(TSHARP i, SSHARP s, float4 c, float l);
extern __attribute__((pure)) half4 OCKL_MANGLE_T(image_sampleh_lod,3D)(TSHARP i, SSHARP s, float4 c, float l);

extern __attribute__((const)) int OCKL_MANGLE_T(image_array_size,1Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_array_size,2Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_array_size,2Dad)(TSHARP i);

extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_data_type,1D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_data_type,1Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_data_type,1Db)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_data_type,2D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_data_type,2Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_data_type,2Dad)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_data_type,2Dd)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_data_type,3D)(TSHARP i);

extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_order,1D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_order,1Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_order,1Db)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_order,2D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_order,2Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_order,2Dad)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_order,2Dd)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_channel_order,3D)(TSHARP i);

extern __attribute__((const)) int OCKL_MANGLE_T(image_depth,3D)(TSHARP i);

extern __attribute__((const)) int OCKL_MANGLE_T(image_height,2D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_height,2Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_height,2Dad)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_height,2Dd)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_height,3D)(TSHARP i);

extern __attribute__((const)) int OCKL_MANGLE_T(image_num_mip_levels,1D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_num_mip_levels,1Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_num_mip_levels,2D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_num_mip_levels,2Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_num_mip_levels,2Dad)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_num_mip_levels,2Dd)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_num_mip_levels,3D)(TSHARP i);

extern __attribute__((const)) int OCKL_MANGLE_T(image_width,1D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_width,1Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_width,1Db)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_width,2D)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_width,2Da)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_width,2Dad)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_width,2Dd)(TSHARP i);
extern __attribute__((const)) int OCKL_MANGLE_T(image_width,3D)(TSHARP i);

#endif // OCKL_H

