add_library(
  libc_diff_test_utils
  Timer.cpp
  Timer.h
)

# A convenience target to build all differential tests.
add_custom_target(libc-math-differential-tests)

function(add_diff_binary target_name)
  cmake_parse_arguments(
    "DIFF"
    "" # No optional arguments
    "SUITE;CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT DIFF_SRCS)
    message(FATAL_ERROR "'add_diff_binary' target requires a SRCS list of .cpp "
                        "files.")
  endif()
  if(NOT DIFF_DEPENDS)
    message(FATAL_ERROR "'add_diff_binary' target requires a DEPENDS list of "
                        "'add_entrypoint_object' targets.")
  endif()

  get_fq_target_name(${target_name} fq_target_name)
  get_fq_deps_list(fq_deps_list ${DIFF_DEPENDS})
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
    ${DIFF_SRCS}
    ${DIFF_HDRS}
  )
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_SOURCE_DIR}/include
  )
  if(DIFF_COMPILE_OPTIONS)
    target_compile_options(
      ${fq_target_name}
      PRIVATE ${DIFF_COMPILE_OPTIONS}
    )
  endif()

  target_link_libraries(
      ${fq_target_name}
      PRIVATE ${link_object_files} libc_diff_test_utils)

  set_target_properties(${fq_target_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if(DIFF_CXX_STANDARD)
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        CXX_STANDARD ${DIFF_CXX_STANDARD}
    )
  endif()

  add_dependencies(
    ${fq_target_name}
    libc.src.__support.FPUtil.fp_bits
    ${fq_deps_list}
  )
  add_dependencies(libc-math-differential-tests ${fq_target_name})
endfunction()

add_header_library(
  single_input_single_output_diff
  HDRS
    SingleInputSingleOutputDiff.h
)

add_header_library(
  binary_op_single_output_diff
  HDRS
    BinaryOpSingleOutputDiff.h
)

add_diff_binary(
  sinf_diff
  SRCS
    sinf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.sinf
)

add_diff_binary(
  sinf_perf
  SRCS
    sinf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.sinf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  cosf_diff
  SRCS
    cosf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.cosf
)

add_diff_binary(
  cosf_perf
  SRCS
    cosf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.cosf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  expm1f_diff
  SRCS
    expm1f_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.expm1f
)

add_diff_binary(
  expm1f_perf
  SRCS
    expm1f_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.expm1f
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  ceilf_diff
  SRCS
    ceilf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.ceilf
)

add_diff_binary(
  ceilf_perf
  SRCS
    ceilf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.ceilf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  exp2f_diff
  SRCS
    exp2f_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.exp2f
)

add_diff_binary(
  exp2f_perf
  SRCS
    exp2f_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.exp2f
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  expf_diff
  SRCS
    expf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.expf
)

add_diff_binary(
  expf_perf
  SRCS
    expf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.expf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  fabsf_diff
  SRCS
    fabsf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fabsf
)

add_diff_binary(
  fabsf_perf
  SRCS
    fabsf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fabsf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  floorf_diff
  SRCS
    floorf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.floorf
)

add_diff_binary(
  floorf_perf
  SRCS
    floorf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.floorf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  log10f_perf
  SRCS
    log10f_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.log10f
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  log1pf_perf
  SRCS
    log1pf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.log1pf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  log2f_diff
  SRCS
    log2f_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.log2f
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  log2f_perf
  SRCS
    log2f_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.log2f
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  logf_diff
  SRCS
    logf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.logf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  logf_perf
  SRCS
    logf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.logf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  logbf_diff
  SRCS
    logbf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.logbf
)

add_diff_binary(
  logbf_perf
  SRCS
    logbf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.logbf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  nearbyintf_diff
  SRCS
    nearbyintf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.nearbyintf
)

add_diff_binary(
  nearbyintf_perf
  SRCS
    nearbyintf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.nearbyintf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  rintf_diff
  SRCS
    rintf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.rintf
)

add_diff_binary(
  rintf_perf
  SRCS
    rintf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.rintf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  roundf_diff
  SRCS
    roundf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.roundf
)

add_diff_binary(
  roundf_perf
  SRCS
    roundf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.roundf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  sqrtf_diff
  SRCS
    sqrtf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.sqrtf
)

add_diff_binary(
  sqrtf_perf
  SRCS
    sqrtf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.sqrtf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  truncf_diff
  SRCS
    truncf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.truncf
)

add_diff_binary(
  truncf_perf
  SRCS
    truncf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.truncf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  hypotf_diff
  SRCS
    hypotf_diff.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.hypotf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  hypotf_perf
  SRCS
    hypotf_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.hypotf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  hypot_diff
  SRCS
    hypot_diff.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.hypot
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  hypot_perf
  SRCS
    hypot_perf.cpp
  DEPENDS
    .binary_op_single_output_diff
    libc.src.math.hypot
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  fmodf_diff
  SRCS
    fmodf_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fmodf
)

add_diff_binary(
  fmodf_perf
  SRCS
    fmodf_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fmodf
  COMPILE_OPTIONS
    -fno-builtin
)

add_diff_binary(
  fmod_diff
  SRCS
    fmod_diff.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fmod
)

add_diff_binary(
  fmod_perf
  SRCS
    fmod_perf.cpp
  DEPENDS
    .single_input_single_output_diff
    libc.src.math.fmod
  COMPILE_OPTIONS
    -fno-builtin
)
