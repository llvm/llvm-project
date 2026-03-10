libclc_configure_source_list(CLC_AMDGPU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}
  math/clc_half_exp.cl
  math/clc_half_exp2.cl
  math/clc_half_exp10.cl
  math/clc_half_log.cl
  math/clc_half_log2.cl
  math/clc_half_log10.cl
  math/clc_half_recip.cl
  math/clc_half_rsqrt.cl
  math/clc_half_sqrt.cl
  math/clc_native_exp.cl
  math/clc_native_exp2.cl
  math/clc_native_log10.cl
  math/clc_sqrt.cl
)

libclc_configure_source_options(${CMAKE_CURRENT_SOURCE_DIR} -fapprox-func
  math/clc_native_exp.cl
  math/clc_native_exp2.cl
  math/clc_native_log10.cl
)
