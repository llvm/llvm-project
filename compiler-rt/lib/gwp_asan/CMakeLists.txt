add_compiler_rt_component(gwp_asan)

include_directories(..)

set(GWP_ASAN_SOURCES
  platform_specific/guarded_pool_allocator_posix.cpp
  platform_specific/mutex_posix.cpp
  guarded_pool_allocator.cpp
  random.cpp
  stack_trace_compressor.cpp
)

set(GWP_ASAN_HEADERS
  definitions.h
  guarded_pool_allocator.h
  mutex.h
  options.h
  options.inc
  random.h
  stack_trace_compressor.h
)

# Ensure that GWP-ASan meets the delegated requirements of some supporting
# allocators. Some supporting allocators (e.g. scudo standalone) cannot use any
# parts of the C++ standard library.
set(GWP_ASAN_CFLAGS ${SANITIZER_COMMON_CFLAGS} -fno-rtti -fno-exceptions
    -nostdinc++ -pthread)
append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC GWP_ASAN_CFLAGS)
append_list_if(COMPILER_RT_HAS_OMIT_FRAME_POINTER_FLAG -fno-omit-frame-pointer
               GWP_ASAN_CFLAGS)

# Remove -stdlib= which is unused when passing -nostdinc++.
string(REGEX REPLACE "-stdlib=[a-zA-Z+]*" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# Options parsing support is optional. GwpAsan is totally independent of
# sanitizer_common, the options parser is not. This is an optional library
# that can be used by an allocator to automatically parse GwpAsan options from
# the environment variable GWP_ASAN_FLAGS, but the allocator can choose to
# implement its own options parsing and populate the Options struct itself.
set(GWP_ASAN_OPTIONS_PARSER_SOURCES
  optional/options_parser.cpp
)
set(GWP_ASAN_OPTIONS_PARSER_HEADERS
  optional/options_parser.h
  options.h
  options.inc
)
set(GWP_ASAN_BACKTRACE_HEADERS
  optional/backtrace.h
  options.h
  options.inc
)

set(GWP_ASAN_OPTIONS_PARSER_CFLAGS
    ${GWP_ASAN_CFLAGS}
    ${SANITIZER_COMMON_CFLAGS})
set(GWP_ASAN_OPTIONS_PARSER_OBJECT_LIBS
    RTSanitizerCommon
    RTSanitizerCommonNoLibc)

if (COMPILER_RT_HAS_GWP_ASAN)
  foreach(arch ${GWP_ASAN_SUPPORTED_ARCH})
    add_compiler_rt_runtime(
      clang_rt.gwp_asan
      STATIC
      ARCHS ${arch}
      SOURCES ${GWP_ASAN_SOURCES}
      ADDITIONAL_HEADERS ${GWP_ASAN_HEADERS}
      CFLAGS ${GWP_ASAN_CFLAGS}
      PARENT_TARGET gwp_asan
    )
  endforeach()

  add_compiler_rt_object_libraries(RTGwpAsan
      ARCHS ${GWP_ASAN_SUPPORTED_ARCH}
      SOURCES ${GWP_ASAN_SOURCES}
      ADDITIONAL_HEADERS ${GWP_ASAN_HEADERS}
      CFLAGS ${GWP_ASAN_CFLAGS})

  # Note: If you choose to add this as an object library, ensure you also
  # include the sanitizer_common flag parsing object lib (generally
  # 'RTSanitizerCommonNoTermination'). Also, you'll need to either implement
  # your own backtrace support (see optional/backtrace.h), or choose between one
  # of the pre-implemented backtrace support options (see below).
  add_compiler_rt_object_libraries(RTGwpAsanOptionsParser
      ARCHS ${GWP_ASAN_SUPPORTED_ARCH}
      SOURCES ${GWP_ASAN_OPTIONS_PARSER_SOURCES}
      ADDITIONAL_HEADERS ${GWP_ASAN_OPTIONS_PARSER_HEADERS}
      CFLAGS ${GWP_ASAN_OPTIONS_PARSER_CFLAGS})

  # As above, build the pre-implemented optional backtrace support libraries.
  add_compiler_rt_object_libraries(RTGwpAsanBacktraceLibc
      ARCHS ${GWP_ASAN_SUPPORTED_ARCH}
      SOURCES optional/backtrace_linux_libc.cpp
      ADDITIONAL_HEADERS ${GWP_ASAN_BACKTRACE_HEADERS}
      CFLAGS ${GWP_ASAN_CFLAGS})
  add_compiler_rt_object_libraries(RTGwpAsanBacktraceSanitizerCommon
      ARCHS ${GWP_ASAN_SUPPORTED_ARCH}
      SOURCES optional/backtrace_sanitizer_common.cpp
      ADDITIONAL_HEADERS ${GWP_ASAN_BACKTRACE_HEADERS}
      CFLAGS ${GWP_ASAN_CFLAGS} ${SANITIZER_COMMON_CFLAGS})
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
