#------------------------------------------------------------------------------
# Cpu features definition and flags
#
# Declare a list of all supported cpu features in ALL_CPU_FEATURES.
#
# Declares associated flags to enable/disable individual feature of the form:
# - CPU_FEATURE_<FEATURE>_ENABLE_FLAG
# - CPU_FEATURE_<FEATURE>_DISABLE_FLAG
#
#------------------------------------------------------------------------------

if(${LIBC_TARGET_MACHINE} MATCHES "x86|x86_64")
  set(ALL_CPU_FEATURES SSE SSE2 AVX AVX512F)
endif()

function(_define_cpu_feature_flags feature)
  if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    string(TOLOWER ${feature} lowercase_feature)
    set(CPU_FEATURE_${feature}_ENABLE_FLAG "-m${lowercase_feature}" PARENT_SCOPE)
    set(CPU_FEATURE_${feature}_DISABLE_FLAG "-mno-${lowercase_feature}" PARENT_SCOPE)
  else()
    # In future, we can extend for other compilers.
    message(FATAL_ERROR "Unkown compiler ${CMAKE_CXX_COMPILER_ID}.")
  endif()
endfunction()

# Defines cpu features flags
foreach(feature IN LISTS ALL_CPU_FEATURES)
  _define_cpu_feature_flags(${feature})
endforeach()

#------------------------------------------------------------------------------
# Optimization level flags
#
# Generates the set of flags needed to compile for a up to a particular
# optimization level.
#
# Creates variables of the form `CPU_FEATURE_OPT_<FEATURE>_FLAGS`.
# CPU_FEATURE_OPT_NONE_FLAGS is a special flag for which no feature is needed.
#
# e.g.
# CPU_FEATURE_OPT_NONE_FLAGS : -mno-sse;-mno-sse2;-mno-avx;-mno-avx512f
# CPU_FEATURE_OPT_SSE_FLAGS : -msse;-mno-sse2;-mno-avx;-mno-avx512f
# CPU_FEATURE_OPT_SSE2_FLAGS : -msse;-msse2;-mno-avx;-mno-avx512f
# CPU_FEATURE_OPT_AVX_FLAGS : -msse;-msse2;-mavx;-mno-avx512f
# CPU_FEATURE_OPT_AVX512F_FLAGS : -msse;-msse2;-mavx;-mavx512f
#------------------------------------------------------------------------------

# Helper function to concatenate flags needed to support optimization up to
# a particular feature.
function(_generate_flags_for_up_to feature flag_variable)
  list(FIND ALL_CPU_FEATURES ${feature} feature_index)
  foreach(current_feature IN LISTS ALL_CPU_FEATURES)
    list(FIND ALL_CPU_FEATURES ${current_feature} current_feature_index)  
    if(${current_feature_index} GREATER ${feature_index})
      list(APPEND flags ${CPU_FEATURE_${current_feature}_DISABLE_FLAG})
    else()
      list(APPEND flags ${CPU_FEATURE_${current_feature}_ENABLE_FLAG})
    endif()
  endforeach()
  set(${flag_variable} ${flags} PARENT_SCOPE)
endfunction()

function(_generate_opt_levels)
  set(opt_levels NONE)
  list(APPEND opt_levels ${ALL_CPU_FEATURES})
  foreach(feature IN LISTS opt_levels)
    set(flag_name "CPU_FEATURE_OPT_${feature}_FLAGS")
    _generate_flags_for_up_to(${feature} ${flag_name})
    set(${flag_name} ${${flag_name}} PARENT_SCOPE)
  endforeach()
endfunction()

_generate_opt_levels()

#------------------------------------------------------------------------------
# Host cpu feature introspection
#
# Populates a HOST_CPU_FEATURES list containing the available CPU_FEATURE.
#------------------------------------------------------------------------------
function(_check_host_cpu_feature feature)
  string(TOLOWER ${feature} lowercase_feature)
  try_run(
    run_result
    compile_result
    "${CMAKE_CURRENT_BINARY_DIR}/check_${lowercase_feature}"
    "${CMAKE_MODULE_PATH}/cpu_features/check_${lowercase_feature}.cpp"
    COMPILE_DEFINITIONS ${CPU_FEATURE_${feature}_ENABLE_FLAG}
    OUTPUT_VARIABLE compile_output
  )
  if(${compile_result} AND ("${run_result}" EQUAL 0))
    list(APPEND HOST_CPU_FEATURES ${feature})
    set(HOST_CPU_FEATURES ${HOST_CPU_FEATURES} PARENT_SCOPE)
  endif()
endfunction()

foreach(feature IN LISTS ALL_CPU_FEATURES)
  _check_host_cpu_feature(${feature})
endforeach()
