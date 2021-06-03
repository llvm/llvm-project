add_compiler_rt_component(scudo_standalone)

include_directories(../.. include)

set(SCUDO_CFLAGS)

list(APPEND SCUDO_CFLAGS
  -Werror=conversion
  -Wall
  -g
  -nostdinc++)

# Remove -stdlib= which is unused when passing -nostdinc++.
string(REGEX REPLACE "-stdlib=[a-zA-Z+]*" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

append_list_if(COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG -fvisibility=hidden SCUDO_CFLAGS)

append_list_if(COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions SCUDO_CFLAGS)

append_list_if(COMPILER_RT_HAS_WNO_PEDANTIC -Wno-pedantic SCUDO_CFLAGS)

# FIXME: find cleaner way to agree with GWPAsan flags
append_list_if(COMPILER_RT_HAS_FNO_LTO_FLAG -fno-lto SCUDO_CFLAGS)

if(COMPILER_RT_DEBUG)
  list(APPEND SCUDO_CFLAGS -O0 -DSCUDO_DEBUG=1)
else()
  list(APPEND SCUDO_CFLAGS -O3)
endif()

set(SCUDO_LINK_FLAGS)

list(APPEND SCUDO_LINK_FLAGS -Wl,-z,defs,-z,now,-z,relro)

list(APPEND SCUDO_LINK_FLAGS -ffunction-sections -fdata-sections -Wl,--gc-sections)

# We don't use the C++ standard library, so avoid including it by mistake.
append_list_if(COMPILER_RT_HAS_NOSTDLIBXX_FLAG -nostdlib++ SCUDO_LINK_FLAGS)

if(ANDROID)
  list(APPEND SCUDO_CFLAGS -fno-emulated-tls)

# Put the shared library in the global group. For more details, see
# android-changes-for-ndk-developers.md#changes-to-library-search-order
  append_list_if(COMPILER_RT_HAS_Z_GLOBAL -Wl,-z,global SCUDO_LINK_FLAGS)
endif()

set(SCUDO_HEADERS
  allocator_config.h
  atomic_helpers.h
  bytemap.h
  checksum.h
  chunk.h
  combined.h
  common.h
  flags_parser.h
  flags.h
  fuchsia.h
  internal_defs.h
  linux.h
  list.h
  local_cache.h
  memtag.h
  mutex.h
  options.h
  platform.h
  primary32.h
  primary64.h
  quarantine.h
  release.h
  report.h
  secondary.h
  size_class_map.h
  stack_depot.h
  stats.h
  string_utils.h
  tsd_exclusive.h
  tsd_shared.h
  tsd.h
  vector.h
  wrappers_c_checks.h
  wrappers_c.h

  include/scudo/interface.h
  )

set(SCUDO_SOURCES
  checksum.cpp
  common.cpp
  crc32_hw.cpp
  flags_parser.cpp
  flags.cpp
  fuchsia.cpp
  linux.cpp
  release.cpp
  report.cpp
  string_utils.cpp
  )

# Enable the SSE 4.2 instruction set for crc32_hw.cpp, if available.
if (COMPILER_RT_HAS_MSSE4_2_FLAG)
  set_source_files_properties(crc32_hw.cpp PROPERTIES COMPILE_FLAGS -msse4.2)
endif()

# Enable the AArch64 CRC32 feature for crc32_hw.cpp, if available.
# Note that it is enabled by default starting with armv8.1-a.
if (COMPILER_RT_HAS_MCRC_FLAG)
  set_source_files_properties(crc32_hw.cpp PROPERTIES COMPILE_FLAGS -mcrc)
endif()

set(SCUDO_SOURCES_C_WRAPPERS
  wrappers_c.cpp
  )

set(SCUDO_SOURCES_CXX_WRAPPERS
  wrappers_cpp.cpp
  )

set(SCUDO_OBJECT_LIBS)

if (COMPILER_RT_HAS_GWP_ASAN)
  add_dependencies(scudo_standalone gwp_asan)
  list(APPEND SCUDO_OBJECT_LIBS
       RTGwpAsan RTGwpAsanBacktraceLibc RTGwpAsanSegvHandler
       RTGwpAsanOptionsParser)

  list(APPEND SCUDO_CFLAGS -DGWP_ASAN_HOOKS -fno-omit-frame-pointer
                           -mno-omit-leaf-frame-pointer)
endif()

set(SCUDO_LINK_LIBS)

append_list_if(COMPILER_RT_HAS_LIBPTHREAD -pthread SCUDO_LINK_FLAGS)

append_list_if(FUCHSIA zircon SCUDO_LINK_LIBS)

if(COMPILER_RT_HAS_SCUDO_STANDALONE)
  add_compiler_rt_object_libraries(RTScudoStandalone
    ARCHS ${SCUDO_STANDALONE_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS})
  add_compiler_rt_object_libraries(RTScudoStandaloneCWrappers
    ARCHS ${SCUDO_STANDALONE_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES_C_WRAPPERS}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS})
  add_compiler_rt_object_libraries(RTScudoStandaloneCxxWrappers
    ARCHS ${SCUDO_STANDALONE_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES_CXX_WRAPPERS}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS})

  add_compiler_rt_runtime(clang_rt.scudo_standalone
    STATIC
    ARCHS ${SCUDO_STANDALONE_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES} ${SCUDO_SOURCES_C_WRAPPERS}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS}
    OBJECT_LIBS ${SCUDO_OBJECT_LIBS}
    PARENT_TARGET scudo_standalone)
  add_compiler_rt_runtime(clang_rt.scudo_standalone_cxx
    STATIC
    ARCHS ${SCUDO_STANDALONE_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES_CXX_WRAPPERS}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS}
    PARENT_TARGET scudo_standalone)

  add_compiler_rt_runtime(clang_rt.scudo_standalone
    SHARED
    ARCHS ${SCUDO_STANDALONE_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES} ${SCUDO_SOURCES_C_WRAPPERS} ${SCUDO_SOURCES_CXX_WRAPPERS}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS}
    OBJECT_LIBS ${SCUDO_OBJECT_LIBS}
    LINK_FLAGS ${SCUDO_LINK_FLAGS}
    LINK_LIBS ${SCUDO_LINK_LIBS}
    PARENT_TARGET scudo_standalone)

  add_subdirectory(benchmarks)
  if(COMPILER_RT_INCLUDE_TESTS)
    add_subdirectory(tests)
  endif()
endif()
