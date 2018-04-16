add_compiler_rt_component(scudo)

include_directories(..)

set(SCUDO_CFLAGS ${SANITIZER_COMMON_CFLAGS})
# SANITIZER_COMMON_CFLAGS include -fno-builtin, but we actually want builtins!
list(APPEND SCUDO_CFLAGS -fbuiltin)
append_rtti_flag(OFF SCUDO_CFLAGS)

set(SCUDO_DYNAMIC_LIBS ${SANITIZER_COMMON_LINK_LIBS})
append_list_if(COMPILER_RT_HAS_LIBDL dl SCUDO_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt SCUDO_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread SCUDO_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBLOG log SCUDO_DYNAMIC_LIBS)

set(SCUDO_DYNAMIC_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})
# Use gc-sections by default to avoid unused code being pulled in.
list(APPEND SCUDO_DYNAMIC_LINK_FLAGS -Wl,--gc-sections)

set(SCUDO_OBJECT_LIBS
  RTSanitizerCommonNoTermination
  RTSanitizerCommonLibc
  RTInterception)

if (FUCHSIA)
  list(APPEND SCUDO_CFLAGS -nostdinc++)
  list(APPEND SCUDO_DYNAMIC_LINK_FLAGS -nostdlib++)
else()
  list(APPEND SCUDO_OBJECT_LIBS
    RTSanitizerCommonCoverage
    RTSanitizerCommonSymbolizer
    RTUbsan)
  list(APPEND SCUDO_DYNAMIC_LIBS ${SANITIZER_CXX_ABI_LIBRARY})
endif()

set(SCUDO_SOURCES
  scudo_allocator.cpp
  scudo_crc32.cpp
  scudo_flags.cpp
  scudo_malloc.cpp
  scudo_termination.cpp
  scudo_tsd_exclusive.cpp
  scudo_tsd_shared.cpp
  scudo_utils.cpp)

set(SCUDO_CXX_SOURCES
  scudo_new_delete.cpp)

# Enable the SSE 4.2 instruction set for scudo_crc32.cpp, if available.
if (COMPILER_RT_HAS_MSSE4_2_FLAG)
  set_source_files_properties(scudo_crc32.cpp PROPERTIES COMPILE_FLAGS -msse4.2)
endif()

# Enable the AArch64 CRC32 feature for scudo_crc32.cpp, if available.
# Note that it is enabled by default starting with armv8.1-a.
if (COMPILER_RT_HAS_MCRC_FLAG)
  set_source_files_properties(scudo_crc32.cpp PROPERTIES COMPILE_FLAGS -mcrc)
endif()

if(COMPILER_RT_HAS_SCUDO)
  add_compiler_rt_runtime(clang_rt.scudo
    STATIC
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES}
    OBJECT_LIBS ${SCUDO_OBJECT_LIBS}
    CFLAGS ${SCUDO_CFLAGS}
    PARENT_TARGET scudo)

  add_compiler_rt_runtime(clang_rt.scudo_cxx
    STATIC
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_CXX_SOURCES}
    OBJECT_LIBS RTUbsan_cxx
    CFLAGS ${SCUDO_CFLAGS}
    PARENT_TARGET scudo)

  add_compiler_rt_runtime(clang_rt.scudo
    SHARED
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES} ${SCUDO_CXX_SOURCES}
    OBJECT_LIBS ${SCUDO_OBJECT_LIBS}
    CFLAGS ${SCUDO_CFLAGS}
    LINK_FLAGS ${SCUDO_DYNAMIC_LINK_FLAGS}
    LINK_LIBS ${SCUDO_DYNAMIC_LIBS}
    PARENT_TARGET scudo)
endif()
