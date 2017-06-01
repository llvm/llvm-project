# First, add the subdirectories which contain feature-based runtime libraries
# and several convenience helper libraries.

include(AddCompilerRT)
include(SanitizerUtils)

# Hoist the building of sanitizer_common on whether we're building either the
# sanitizers or xray (or both).
#
#TODO: Refactor sanitizer_common into smaller pieces (e.g. flag parsing, utils).
if (COMPILER_RT_HAS_SANITIZER_COMMON AND
    (COMPILER_RT_BUILD_SANITIZERS OR COMPILER_RT_BUILD_XRAY))
  add_subdirectory(sanitizer_common)
endif()

if(COMPILER_RT_BUILD_BUILTINS)
  add_subdirectory(builtins)
endif()

function(compiler_rt_build_runtime runtime)
  string(TOUPPER ${runtime} runtime_uppercase)
  if(COMPILER_RT_HAS_${runtime_uppercase})
    add_subdirectory(${runtime})
    foreach(directory ${ARGN})
      add_subdirectory(${directory})
    endforeach()
  endif()
endfunction()

function(compiler_rt_build_sanitizer sanitizer)
  string(TOUPPER ${sanitizer} sanitizer_uppercase)
  string(TOLOWER ${sanitizer} sanitizer_lowercase)
  list(FIND COMPILER_RT_SANITIZERS_TO_BUILD ${sanitizer_lowercase} result)
  if(NOT ${result} EQUAL -1)
    compiler_rt_build_runtime(${sanitizer} ${ARGN})
  endif()
endfunction()

if(COMPILER_RT_BUILD_SANITIZERS)
  compiler_rt_build_runtime(interception)

  if(COMPILER_RT_HAS_SANITIZER_COMMON)
    add_subdirectory(stats)
    add_subdirectory(lsan)
    add_subdirectory(ubsan)
  endif()

  compiler_rt_build_sanitizer(asan)
  compiler_rt_build_sanitizer(dfsan)
  compiler_rt_build_sanitizer(msan)
  compiler_rt_build_sanitizer(tsan tsan/dd)
  compiler_rt_build_sanitizer(safestack)
  compiler_rt_build_sanitizer(cfi)
  compiler_rt_build_sanitizer(esan)
  compiler_rt_build_sanitizer(scudo)

  compiler_rt_build_runtime(profile)
endif()

if(COMPILER_RT_BUILD_XRAY)
  compiler_rt_build_runtime(xray)
endif()
