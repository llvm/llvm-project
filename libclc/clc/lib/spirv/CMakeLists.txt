if(LIBCLC_TARGET_OS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBCLC_TARGET_OS})
  add_subdirectory(${LIBCLC_TARGET_OS})
endif()

libclc_add_sources(${LIBCLC_CLC_TARGET} FILES
  math/clc_fmax.cl
  math/clc_fmin.cl
  subnormal_config.cl
)
