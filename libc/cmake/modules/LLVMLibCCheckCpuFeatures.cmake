# ------------------------------------------------------------------------------
# Cpu features definition and flags
# ------------------------------------------------------------------------------

if(${LIBC_TARGET_MACHINE} MATCHES "x86|x86_64")
  set(ALL_CPU_FEATURES SSE SSE2 AVX AVX2 AVX512F)
  list(SORT ALL_CPU_FEATURES)
endif()

# Function to check whether the host supports the provided set of features.
# Usage:
# host_supports(
#   <output variable>
#   <list of cpu features>
# )
function(host_supports output_var features)
  _intersection(a "${HOST_CPU_FEATURES}" "${features}")
  if("${a}" STREQUAL "${features}")
    set(${output_var} TRUE PARENT_SCOPE)
  else()
    unset(${output_var} PARENT_SCOPE)
  endif()
endfunction()

# Function to compute the flags to pass down to the compiler.
# Usage:
# compute_flags(
#   <output variable>
#   MARCH <arch name or "native">
#   REQUIRE <list of mandatory features to enable>
#   REJECT <list of features to disable>
# )
function(compute_flags output_var)
  cmake_parse_arguments(
    "COMPUTE_FLAGS"
    "" # Optional arguments
    "MARCH" # Single value arguments
    "REQUIRE;REJECT" # Multi value arguments
    ${ARGN})
  # Check that features are not required and rejected at the same time.
  if(COMPUTE_FLAGS_REQUIRE AND COMPUTE_FLAGS_REJECT)
    _intersection(var ${COMPUTE_FLAGS_REQUIRE} ${COMPUTE_FLAGS_REJECT})
    if(var)
      message(FATAL_ERROR "Cpu Features REQUIRE and REJECT ${var}")
    endif()
  endif()
  # Generate the compiler flags in `current`.
  if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang|GNU")
    if(COMPUTE_FLAGS_MARCH)
      list(APPEND current "-march=${COMPUTE_FLAGS_MARCH}")
    endif()
    foreach(feature IN LISTS COMPUTE_FLAGS_REQUIRE)
      string(TOLOWER ${feature} lowercase_feature)
      list(APPEND current "-m${lowercase_feature}")
    endforeach()
    foreach(feature IN LISTS COMPUTE_FLAGS_REJECT)
      string(TOLOWER ${feature} lowercase_feature)
      list(APPEND current "-mno-${lowercase_feature}")
    endforeach()
  else()
    # In future, we can extend for other compilers.
    message(FATAL_ERROR "Unkown compiler ${CMAKE_CXX_COMPILER_ID}.")
  endif()
  # Export the list of flags.
  set(${output_var} "${current}" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------
# Internal helpers and utilities.
# ------------------------------------------------------------------------------

# Computes the intersection between two lists.
function(_intersection output_var list1 list2)
  foreach(element IN LISTS list1)
    if("${list2}" MATCHES "(^|;)${element}(;|$)")
      list(APPEND tmp "${element}")
    endif()
  endforeach()
  set(${output_var} ${tmp} PARENT_SCOPE)
endfunction()

# Generates a cpp file to introspect the compiler defined flags.
function(_generate_check_code)
  foreach(feature IN LISTS ALL_CPU_FEATURES)
    set(DEFINITIONS
        "${DEFINITIONS}
#ifdef __${feature}__
    \"${feature}\",
#endif")
  endforeach()
  configure_file(
    "${LIBC_SOURCE_DIR}/cmake/modules/cpu_features/check_cpu_features.cpp.in"
    "cpu_features/check_cpu_features.cpp" @ONLY)
endfunction()
_generate_check_code()

# Compiles and runs the code generated above with the specified requirements.
# This is helpful to infer which features a particular target supports or if
# a specific features implies other features (e.g. BMI2 implies SSE2 and SSE).
function(_check_defined_cpu_feature output_var)
  cmake_parse_arguments(
    "CHECK_DEFINED"
    "" # Optional arguments
    "MARCH" # Single value arguments
    "REQUIRE;REJECT" # Multi value arguments
    ${ARGN})
  compute_flags(
    flags
    MARCH  ${CHECK_DEFINED_MARCH}
    REQUIRE ${CHECK_DEFINED_REQUIRE}
    REJECT  ${CHECK_DEFINED_REJECT})
  try_run(
    run_result compile_result "${CMAKE_CURRENT_BINARY_DIR}/check_${feature}"
    "${CMAKE_CURRENT_BINARY_DIR}/cpu_features/check_cpu_features.cpp"
    COMPILE_DEFINITIONS ${flags}
    COMPILE_OUTPUT_VARIABLE compile_output
    RUN_OUTPUT_VARIABLE run_output)
  if(${compile_result} AND ("${run_result}" EQUAL 0))
    set(${output_var}
        "${run_output}"
        PARENT_SCOPE)
  else()
    message(FATAL_ERROR "${compile_output}")
  endif()
endfunction()

# Populates the HOST_CPU_FEATURES list.
# Use -march=native only when the compiler supports it.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
  _check_defined_cpu_feature(HOST_CPU_FEATURES MARCH native)
else()
  _check_defined_cpu_feature(HOST_CPU_FEATURES)
endif()
