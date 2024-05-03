add_compiler_rt_component(ctx_profile)

set(CTX_PROFILE_SOURCES
  CtxInstrProfiling.cpp
  )

set(CTX_PROFILE_HEADERS
  CtxInstrProfiling.h
  )

include_directories(..)
include_directories(../../include)

# We don't use the C++ Standard Library here, so avoid including it by mistake.
append_list_if(COMPILER_RT_HAS_NOSTDINCXX_FLAG -nostdinc++ EXTRA_FLAGS)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
