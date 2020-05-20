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

if(COMPILER_RT_BUILD_CRT AND COMPILER_RT_HAS_CRT)
  add_subdirectory(crt)
endif()

function(compiler_rt_build_runtime runtime)
  string(TOUPPER ${runtime} runtime_uppercase)
  if(COMPILER_RT_HAS_${runtime_uppercase})
    add_subdirectory(${runtime})
    if(${runtime} STREQUAL tsan)
      add_subdirectory(tsan/dd)
    endif()
    if(${runtime} STREQUAL scudo)
      add_subdirectory(scudo/standalone)
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
endif()

if(COMPILER_RT_BUILD_PROFILE AND COMPILER_RT_HAS_PROFILE)
  compiler_rt_build_runtime(profile)
endif()

if(COMPILER_RT_BUILD_XRAY)
  compiler_rt_build_runtime(xray)
endif()

if(COMPILER_RT_BUILD_LIBFUZZER)
  compiler_rt_build_runtime(fuzzer)
endif()

# It doesn't normally make sense to build runtimes when a sanitizer is enabled,
# so we don't add_subdirectory the runtimes in that case. However, the opposite
# is true for fuzzers that exercise parts of the runtime. So we add the fuzzer
# directories explicitly here.
add_subdirectory(scudo/standalone/fuzz)
