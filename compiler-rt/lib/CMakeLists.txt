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

if(COMPILER_RT_BUILD_SANITIZERS)
  if(COMPILER_RT_HAS_INTERCEPTION)
    add_subdirectory(interception)
  endif()

  if(COMPILER_RT_HAS_SANITIZER_COMMON)
    add_subdirectory(stats)
    add_subdirectory(lsan)
    add_subdirectory(ubsan)
  endif()

  if(COMPILER_RT_HAS_ASAN)
    add_subdirectory(asan)
  endif()

  if(COMPILER_RT_HAS_DFSAN)
    add_subdirectory(dfsan)
  endif()

  if(COMPILER_RT_HAS_MSAN)
    add_subdirectory(msan)
  endif()

  if(COMPILER_RT_HAS_PROFILE)
    add_subdirectory(profile)
  endif()

  if(COMPILER_RT_HAS_TSAN)
    add_subdirectory(tsan)
    add_subdirectory(tsan/dd)
  endif()

  if(COMPILER_RT_HAS_SAFESTACK)
    add_subdirectory(safestack)
  endif()

  if(COMPILER_RT_HAS_CFI)
    add_subdirectory(cfi)
  endif()

  if(COMPILER_RT_HAS_ESAN)
    add_subdirectory(esan)
  endif()

  if(COMPILER_RT_HAS_SCUDO)
    add_subdirectory(scudo)
  endif()
endif()

if(COMPILER_RT_BUILD_XRAY AND COMPILER_RT_HAS_XRAY)
  add_subdirectory(xray)
endif()
