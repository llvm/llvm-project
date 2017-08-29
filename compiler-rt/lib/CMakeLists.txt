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
    if(${runtime} STREQUAL tsan)
      add_subdirectory(tsan/dd)
    endif()
  endif()
endfunction()

if(COMPILER_RT_BUILD_SANITIZERS)
  compiler_rt_build_runtime(interception)

  if(COMPILER_RT_HAS_SANITIZER_COMMON)
    add_subdirectory(stats)
    add_subdirectory(lsan)
    add_subdirectory(ubsan)
  endif()

  foreach(sanitizer ${COMPILER_RT_SANITIZERS_TO_BUILD})
    compiler_rt_build_runtime(${sanitizer})
  endforeach()

  compiler_rt_build_runtime(profile)
endif()

if(COMPILER_RT_BUILD_XRAY)
  compiler_rt_build_runtime(xray)
endif()

if(COMPILER_RT_BUILD_LIBFUZZER)
  compiler_rt_build_runtime(fuzzer)
endif()
