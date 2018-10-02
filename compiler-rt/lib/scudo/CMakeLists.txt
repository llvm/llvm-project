add_compiler_rt_component(scudo)

include_directories(..)

set(SCUDO_CFLAGS ${SANITIZER_COMMON_CFLAGS})
# SANITIZER_COMMON_CFLAGS include -fno-builtin, but we actually want builtins!
list(APPEND SCUDO_CFLAGS -fbuiltin)
append_rtti_flag(OFF SCUDO_CFLAGS)

set(SCUDO_MINIMAL_DYNAMIC_LIBS ${SANITIZER_COMMON_LINK_LIBS})
append_list_if(COMPILER_RT_HAS_LIBDL dl SCUDO_MINIMAL_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt SCUDO_MINIMAL_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread SCUDO_MINIMAL_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBLOG log SCUDO_MINIMAL_DYNAMIC_LIBS)

set(SCUDO_DYNAMIC_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})
# Use gc-sections by default to avoid unused code being pulled in.
list(APPEND SCUDO_DYNAMIC_LINK_FLAGS -Wl,--gc-sections)

if(ANDROID)
# Put most Sanitizer shared libraries in the global group. For more details, see
# android-changes-for-ndk-developers.md#changes-to-library-search-order
  if (COMPILER_RT_HAS_Z_GLOBAL)
    list(APPEND SCUDO_DYNAMIC_LINK_FLAGS -Wl,-z,global)
  endif()
endif()

# The minimal Scudo runtime does not inlude the UBSan runtime.
set(SCUDO_MINIMAL_OBJECT_LIBS
  RTSanitizerCommonNoTermination
  RTSanitizerCommonLibc
  RTInterception)
set(SCUDO_OBJECT_LIBS ${SCUDO_MINIMAL_OBJECT_LIBS})
set(SCUDO_DYNAMIC_LIBS ${SCUDO_MINIMAL_DYNAMIC_LIBS})

if (FUCHSIA)
  list(APPEND SCUDO_CFLAGS -nostdinc++)
  list(APPEND SCUDO_DYNAMIC_LINK_FLAGS -nostdlib++)
else()
  list(APPEND SCUDO_DYNAMIC_LIBS ${SANITIZER_CXX_ABI_LIBRARY})
  list(APPEND SCUDO_OBJECT_LIBS
    RTSanitizerCommonCoverage
    RTSanitizerCommonSymbolizer
    RTUbsan)
endif()

set(SCUDO_SOURCES
  scudo_allocator.cpp
  scudo_crc32.cpp
  scudo_errors.cpp
  scudo_flags.cpp
  scudo_malloc.cpp
  scudo_termination.cpp
  scudo_tsd_exclusive.cpp
  scudo_tsd_shared.cpp
  scudo_utils.cpp)

set(SCUDO_CXX_SOURCES
  scudo_new_delete.cpp)

set(SCUDO_HEADERS
  scudo_allocator.h
  scudo_allocator_combined.h
  scudo_allocator_secondary.h
  scudo_crc32.h
  scudo_errors.h
  scudo_flags.h
  scudo_flags.inc
  scudo_interface_internal.h
  scudo_platform.h
  scudo_tsd.h
  scudo_tsd_exclusive.inc
  scudo_tsd_shared.inc
  scudo_utils.h)

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
  add_compiler_rt_runtime(clang_rt.scudo_minimal
    STATIC
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    OBJECT_LIBS ${SCUDO_MINIMAL_OBJECT_LIBS}
    CFLAGS ${SCUDO_CFLAGS}
    PARENT_TARGET scudo)
  add_compiler_rt_runtime(clang_rt.scudo_cxx_minimal
    STATIC
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_CXX_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS}
    PARENT_TARGET scudo)

  add_compiler_rt_runtime(clang_rt.scudo
    STATIC
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    OBJECT_LIBS ${SCUDO_OBJECT_LIBS}
    CFLAGS ${SCUDO_CFLAGS}
    PARENT_TARGET scudo)
  add_compiler_rt_runtime(clang_rt.scudo_cxx
    STATIC
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_CXX_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    OBJECT_LIBS RTUbsan_cxx
    CFLAGS ${SCUDO_CFLAGS}
    PARENT_TARGET scudo)

  add_compiler_rt_runtime(clang_rt.scudo_minimal
    SHARED
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES} ${SCUDO_CXX_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    OBJECT_LIBS ${SCUDO_MINIMAL_OBJECT_LIBS}
    CFLAGS ${SCUDO_CFLAGS}
    LINK_FLAGS ${SCUDO_DYNAMIC_LINK_FLAGS}
    LINK_LIBS ${SCUDO_MINIMAL_DYNAMIC_LIBS}
    PARENT_TARGET scudo)

  add_compiler_rt_runtime(clang_rt.scudo
    SHARED
    ARCHS ${SCUDO_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES} ${SCUDO_CXX_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    OBJECT_LIBS ${SCUDO_OBJECT_LIBS}
    CFLAGS ${SCUDO_CFLAGS}
    LINK_FLAGS ${SCUDO_DYNAMIC_LINK_FLAGS}
    LINK_LIBS ${SCUDO_DYNAMIC_LIBS}
    PARENT_TARGET scudo)
endif()
