add_library(
  libc_diff_test_utils STATIC
  Timer.cpp
  Timer.h
)
target_include_directories(
  libc_diff_test_utils
  PRIVATE
    ${LIBC_SOURCE_DIR}
)
add_dependencies(
  libc_diff_test_utils
  libc.src.__support.macros.config
)

# A convenience target to build all performance tests.
add_custom_target(libc-math-performance-tests)

function(add_perf_binary target_name)
  cmake_parse_arguments(
    "PERF"
    "" # No optional arguments
    "SUITE;CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;LINK_LIBRARIES" # Multi-value arguments
    ${ARGN}
  )
  if(NOT PERF_SRCS)
    message(FATAL_ERROR "'add_perf_binary' target requires a SRCS list of .cpp "
                        "files.")
  endif()
  if(NOT PERF_DEPENDS)
    message(FATAL_ERROR "'add_perf_binary' target requires a DEPENDS list of "
                        "'add_entrypoint_object' targets.")
  endif()

  get_fq_target_name(${target_name} fq_target_name)
  get_fq_deps_list(fq_deps_list ${PERF_DEPENDS})
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    if(LIBC_CMAKE_VERBOSE_LOGGING)
      set(msg "Will not build ${fq_target_name} as it has missing deps: "
              "${skipped_entrypoints_list}.")
      message(STATUS ${msg})
    endif()
    return()
  endif()

  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${PERF_SRCS}
    ${PERF_HDRS}
  )
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
  )
  if(PERF_COMPILE_OPTIONS)
    target_compile_options(
      ${fq_target_name}
      PRIVATE ${PERF_COMPILE_OPTIONS}
    )
  endif()

  set(link_libraries ${link_object_files})
  foreach(lib IN LISTS PERF_LINK_LIBRARIES)
    list(APPEND link_libraries ${lib}.unit)
  endforeach()
  target_link_libraries(
      ${fq_target_name}
      PRIVATE ${link_libraries} libc_diff_test_utils)

  set_target_properties(${fq_target_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if(PERF_CXX_STANDARD)
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        CXX_STANDARD ${PERF_CXX_STANDARD}
    )
  endif()

  add_dependencies(
    ${fq_target_name}
    libc.src.__support.FPUtil.fp_bits
    ${fq_deps_list}
  )
  add_dependencies(libc-math-performance-tests ${fq_target_name})
endfunction()

add_header_library(
  single_input_single_output_diff
  HDRS
    SingleInputSingleOutputPerf.h
  DEPENDS
    libc.src.__support.CPP.algorithm
    libc.src.__support.FPUtil.fp_bits
)

add_header_library(
  binary_op_single_output_diff
  HDRS
    BinaryOpSingleOutputPerf.h
  DEPENDS
    libc.src.__support.CPP.algorithm
    libc.src.__support.FPUtil.fp_bits
)

add_perf_binary(
  sinf_perf
  SRCS
    sinf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.sinf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  cosf_perf
  SRCS
    cosf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.cosf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  expm1f_perf
  SRCS
    expm1f_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.expm1f
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  ceilf_perf
  SRCS
    ceilf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.ceilf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  exp10f16_perf
  SRCS
    exp10f16_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.exp10f16
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  exp2f_perf
  SRCS
    exp2f_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.exp2f
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  exp2f16_perf
  SRCS
    exp2f16_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.exp2f16
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  expf_perf
  SRCS
    expf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.expf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  expf16_perf
  SRCS
    expf16_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.expf16
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  fabsf_perf
  SRCS
    fabsf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fabsf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  floorf_perf
  SRCS
    floorf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.floorf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  log10f_perf
  SRCS
    log10f_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.log10f
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  log1pf_perf
  SRCS
    log1pf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.log1pf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  log2f_perf
  SRCS
    log2f_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.log2f
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  logf_perf
  SRCS
    logf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.logf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  logbf_perf
  SRCS
    logbf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.logbf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  nearbyintf_perf
  SRCS
    nearbyintf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.nearbyintf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  rintf_perf
  SRCS
    rintf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.rintf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  roundf_perf
  SRCS
    roundf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.roundf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  sqrtf_perf
  SRCS
    sqrtf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.sqrtf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  truncf_perf
  SRCS
    truncf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.truncf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  hypotf_perf
  SRCS
    hypotf_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.hypotf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  hypot_perf
  SRCS
    hypot_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.hypot
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  fmodf_perf
  SRCS
    fmodf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fmodf
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  fmod_perf
  SRCS
    fmod_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fmod
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  fmodl_perf
  SRCS
    fmodl_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fmodl
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  fmodf16_perf
  SRCS
    fmodf16_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.fmodf16
    libc.src.__support.FPUtil.generic.fmod
    libc.src.__support.macros.properties.types
)

add_perf_binary(
  fmodf128_perf
  SRCS
    fmodf128_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fmodf128
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  nearest_integer_funcs_perf
  SRCS
    nearest_integer_funcs_perf.cpp
  DEPENDS
    libc.src.math.ceilf
    libc.src.math.ceilf16
    libc.src.math.floorf
    libc.src.math.floorf16
    libc.src.math.rintf
    libc.src.math.rintf16
    libc.src.math.roundf
    libc.src.math.roundf16
    libc.src.math.roundevenf
    libc.src.math.roundevenf16
    libc.src.math.truncf
    libc.src.math.truncf16
  COMPILE_OPTIONS
    -fno-builtin
  LINK_LIBRARIES
    LibcFPTestHelpers
)

add_perf_binary(
  misc_basic_ops_perf
  SRCS
    misc_basic_ops_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    .single_input_single_output_diff
    libc.src.math.copysignf
    libc.src.math.copysignf16
    libc.src.math.fabsf
    libc.src.math.fabsf16
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  max_min_funcs_perf
  SRCS
    max_min_funcs_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.fmaxf
    libc.src.math.fmaxf16
    libc.src.math.fmaximumf
    libc.src.math.fmaximumf16
    libc.src.math.fmaximum_numf
    libc.src.math.fmaximum_numf16
    libc.src.math.fminf
    libc.src.math.fminf16
    libc.src.math.fminimumf
    libc.src.math.fminimumf16
    libc.src.math.fminimum_numf
    libc.src.math.fminimum_numf16
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  fmul_perf
  SRCS
    fmul_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.fmul
    libc.src.__support.FPUtil.generic.mul
    libc.src.__support.FPUtil.fp_bits
  COMPILE_OPTIONS
    -fno-builtin
)

add_perf_binary(
  fmull_perf
  SRCS
    fmull_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.fmull
  COMPILE_OPTIONS
    -fno-builtin
)
