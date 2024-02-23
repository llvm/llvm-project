# ------------------------------------------------------------------------------
# Compiler features definition and flags
# ------------------------------------------------------------------------------

set(ALL_COMPILER_FEATURES "float128" "fixed_point")

# Making sure ALL_COMPILER_FEATURES is sorted.
list(SORT ALL_COMPILER_FEATURES)

# Function to check whether the compiler supports the provided set of features.
# Usage:
# compiler_supports(
#   <output variable>
#   <list of cpu features>
# )
function(compiler_supports output_var features)
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

set(AVAILABLE_COMPILER_FEATURES "")

# Try compile a C file to check if flag is supported.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
foreach(feature IN LISTS ALL_COMPILER_FEATURES)
  set(compile_options ${LIBC_COMPILE_OPTIONS_NATIVE})
  if(${feature} STREQUAL "fixed_point")
    list(APPEND compile_options "-ffixed-point")
  endif()

  try_compile(
    has_feature
    ${CMAKE_CURRENT_BINARY_DIR}/compiler_features
    SOURCES ${LIBC_SOURCE_DIR}/cmake/modules/compiler_features/check_${feature}.cpp
    COMPILE_DEFINITIONS -I${LIBC_SOURCE_DIR} ${compile_options}
  )
  if(has_feature)
    list(APPEND AVAILABLE_COMPILER_FEATURES ${feature})
    if(${feature} STREQUAL "float128")
      set(LIBC_COMPILER_HAS_FLOAT128 TRUE)
    elseif(${feature} STREQUAL "fixed_point")
      set(LIBC_COMPILER_HAS_FIXED_POINT TRUE)
    endif()
  endif()
endforeach()

message(STATUS "Compiler features available: ${AVAILABLE_COMPILER_FEATURES}")

### Compiler Feature Detection ###

# clang-8+, gcc-12+
check_cxx_compiler_flag("-ftrivial-auto-var-init=pattern" LIBC_CC_SUPPORTS_PATTERN_INIT)

# clang-6+, gcc-13+
check_cxx_compiler_flag("-nostdlib++" LIBC_CC_SUPPORTS_NOSTDLIBPP)
