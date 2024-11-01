# ------------------------------------------------------------------------------
# Cpu features definition and flags
# ------------------------------------------------------------------------------

# Initialize ALL_CPU_FEATURES as empty list.
set(ALL_CPU_FEATURES "")

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  set(ALL_CPU_FEATURES SSE2 SSE4_2 AVX AVX2 AVX512F AVX512BW FMA)
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
  if(LIBC_TARGET_ARCHITECTURE_IS_GPU)
    unset(${output_var} PARENT_SCOPE)
    return()
  endif()
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

set(AVAILABLE_CPU_FEATURES "")
if(LIBC_CROSSBUILD)
  # If we are doing a cross build, we will just assume that all CPU features
  # are available.
  set(AVAILABLE_CPU_FEATURES ${ALL_CPU_FEATURES})
else()
  # Try compile a C file to check if flag is supported.
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  foreach(feature IN LISTS ALL_CPU_FEATURES)
    try_compile(
      has_feature
      ${CMAKE_CURRENT_BINARY_DIR}/cpu_features
      SOURCES ${LIBC_SOURCE_DIR}/cmake/modules/cpu_features/check_${feature}.cpp
      COMPILE_DEFINITIONS -I${LIBC_SOURCE_DIR} ${LIBC_COMPILE_OPTIONS_NATIVE}
    )
    if(has_feature)
      list(APPEND AVAILABLE_CPU_FEATURES ${feature})
    endif()
  endforeach()
endif()

set(LIBC_CPU_FEATURES ${AVAILABLE_CPU_FEATURES} CACHE STRING "Host supported CPU features")

_intersection(cpu_features "${AVAILABLE_CPU_FEATURES}" "${LIBC_CPU_FEATURES}")
if(NOT "${cpu_features}" STREQUAL "${LIBC_CPU_FEATURES}")
  message(FATAL_ERROR "Unsupported CPU features: ${cpu_features}")
else()
  message(STATUS "Set CPU features: ${cpu_features}")
endif()
