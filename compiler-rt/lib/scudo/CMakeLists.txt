add_compiler_rt_component(scudo)

include_directories(..)

set(SCUDO_CFLAGS ${SANITIZER_COMMON_CFLAGS})
# SANITIZER_COMMON_CFLAGS include -fno-builtin, but we actually want builtins!
list(APPEND SCUDO_CFLAGS -fbuiltin)
append_rtti_flag(OFF SCUDO_CFLAGS)

set(SCUDO_SOURCES
  scudo_allocator.cpp
  scudo_flags.cpp
  scudo_crc32.cpp
  scudo_interceptors.cpp
  scudo_new_delete.cpp
  scudo_termination.cpp
  scudo_tls_android.cpp
  scudo_tls_linux.cpp
  scudo_utils.cpp)

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
  foreach(arch ${SCUDO_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.scudo
      STATIC
      ARCHS ${arch}
      SOURCES ${SCUDO_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonNoTermination.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
      CFLAGS ${SCUDO_CFLAGS}
      PARENT_TARGET scudo)
  endforeach()
endif()
