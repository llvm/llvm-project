_CLC_OVERLOAD _CLC_DECL float3 cross(float3 p0, float3 p1);
_CLC_OVERLOAD _CLC_DECL float4 cross(float4 p0, float4 p1);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL double3 cross(double3 p0, double3 p1);
_CLC_OVERLOAD _CLC_DECL double4 cross(double4 p0, double4 p1);
#endif
