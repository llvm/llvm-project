##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../irif/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../oclc/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(native_func_flags -fapprox-func)

# Minimal flags required to enable v_rcp_f16
#
# TODO: Approx func is stronger than required. Need !fpmath 1.0 set on
# the fdiv, but there's no direct control for this now.
set(rcp_f16_flags -freciprocal-math ${native_func_flags})

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/native_rcpH.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/native_rsqrtH.cl
  PROPERTIES COMPILE_FLAGS "${rcp_f16_flags}")

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/native_logF.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/native_expF.cl
  PROPERTIES COMPILE_FLAGS "${native_func_flags}")

opencl_bc_lib(NAME ocml SOURCES ${sources})
