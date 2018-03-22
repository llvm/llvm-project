include_directories(..)

# Runtime library sources and build flags.
set(MSAN_RTL_SOURCES
  msan.cc
  msan_allocator.cc
  msan_chained_origin_depot.cc
  msan_interceptors.cc
  msan_linux.cc
  msan_report.cc
  msan_thread.cc
  msan_poisoning.cc
  )

set(MSAN_RTL_CXX_SOURCES
  msan_new_delete.cc)


set(MSAN_RTL_CFLAGS ${SANITIZER_COMMON_CFLAGS})
if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  append_list_if(COMPILER_RT_HAS_FTLS_MODEL_INITIAL_EXEC -ftls-model=initial-exec MSAN_RTL_CFLAGS)
endif()
append_rtti_flag(OFF MSAN_RTL_CFLAGS)
if(NOT CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  append_list_if(COMPILER_RT_HAS_FPIE_FLAG -fPIE MSAN_RTL_CFLAGS)
endif()
# Prevent clang from generating libc calls.
append_list_if(COMPILER_RT_HAS_FFREESTANDING_FLAG -ffreestanding MSAN_RTL_CFLAGS)

set(MSAN_RUNTIME_LIBRARIES)

# Static runtime library.
add_compiler_rt_component(msan)

foreach(arch ${MSAN_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.msan
    STATIC
    ARCHS ${arch}
    SOURCES ${MSAN_RTL_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
            $<TARGET_OBJECTS:RTUbsan.${arch}>
    CFLAGS ${MSAN_RTL_CFLAGS}
    PARENT_TARGET msan)
  add_compiler_rt_runtime(clang_rt.msan_cxx
    STATIC
    ARCHS ${arch}
    SOURCES ${MSAN_RTL_CXX_SOURCES}
            $<TARGET_OBJECTS:RTUbsan_cxx.${arch}>
    CFLAGS ${MSAN_RTL_CFLAGS}
    PARENT_TARGET msan)
  list(APPEND MSAN_RUNTIME_LIBRARIES clang_rt.msan-${arch}
                                     clang_rt.msan_cxx-${arch})
  if(UNIX)
    add_sanitizer_rt_symbols(clang_rt.msan
      ARCHS ${arch}
      EXTRA msan.syms.extra)
    add_sanitizer_rt_symbols(clang_rt.msan_cxx
      ARCHS ${arch}
      EXTRA msan.syms.extra)
    add_dependencies(msan clang_rt.msan-${arch}-symbols
                          clang_rt.msan_cxx-${arch}-symbols)
  endif()
endforeach()

add_compiler_rt_resource_file(msan_blacklist msan_blacklist.txt msan)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
