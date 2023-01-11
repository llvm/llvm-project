# ------------------------------------------------------------------------------
# Cpu features definition and flags
# ------------------------------------------------------------------------------

# Initialize ALL_CPU_FEATURES as empty list.
set(ALL_CPU_FEATURES "")

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  set(ALL_CPU_FEATURES SSE2 SSE4_2 AVX2 AVX512F AVX512BW FMA)
  set(LIBC_COMPILE_OPTIONS_NATIVE -march=native)
elseif(${LIBC_TARGET_ARCHITECTURE_IS_AARCH64})
  set(LIBC_COMPILE_OPTIONS_NATIVE -mcpu=native)
endif()

# Making sure ALL_CPU_FEATURES is sorted.
list(SORT ALL_CPU_FEATURES)

# Function to check whether the target CPU supports the provided set of features.
# Usage:
# cpu_supports(
#   <output variable>
#   <list of cpu features>
# )
function(cpu_supports output_var features)
  _intersection(var "${LIBC_CPU_FEATURES}" "${features}")
  if("${var}" STREQUAL "${features}")
    set(${output_var} TRUE PARENT_SCOPE)
  else()
    unset(${output_var} PARENT_SCOPE)
  endif()
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

set(LIBC_CPU_FEATURES "" CACHE PATH "Host supported CPU features")

if(LIBC_CROSSBUILD)
  _intersection(cpu_features "${ALL_CPU_FEATURES}" "${LIBC_CPU_FEATURES}")
  if(NOT "${cpu_features}" STREQUAL "${LIBC_CPU_FEATURES}")
    message(FATAL_ERROR "Unsupported CPU features: ${cpu_features}")
  endif()
  message(STATUS "Set CPU features: ${cpu_features}")
  set(LIBC_CPU_FEATURES "${cpu_features}")
else()
  # Populates the LIBC_CPU_FEATURES list from host.
  try_run(
    run_result compile_result "${CMAKE_CURRENT_BINARY_DIR}/check_${feature}"
    "${CMAKE_CURRENT_BINARY_DIR}/cpu_features/check_cpu_features.cpp"
    COMPILE_DEFINITIONS ${LIBC_COMPILE_OPTIONS_NATIVE}
    COMPILE_OUTPUT_VARIABLE compile_output
    RUN_OUTPUT_VARIABLE run_output)
  if("${run_result}" EQUAL 0)
    message(STATUS "Set CPU features: ${run_output}")
    set(LIBC_CPU_FEATURES "${run_output}")
  elseif(NOT ${compile_result})
    message(FATAL_ERROR "Failed to compile: ${compile_output}")
  else()
    message(FATAL_ERROR "Failed to run: ${run_output}")
  endif()
endif()
