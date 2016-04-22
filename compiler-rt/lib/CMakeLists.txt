# First, add the subdirectories which contain feature-based runtime libraries
# and several convenience helper libraries.

include(AddCompilerRT)
include(SanitizerUtils)

if(COMPILER_RT_BUILD_BUILTINS)
  add_subdirectory(builtins)
endif()

if(COMPILER_RT_BUILD_SANITIZERS)
  if(COMPILER_RT_HAS_INTERCEPTION)
    add_subdirectory(interception)
  endif()

  if(COMPILER_RT_HAS_SANITIZER_COMMON)
    add_subdirectory(sanitizer_common)
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
endif()
