include_directories(..)

set(LSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF LSAN_CFLAGS)

set(LSAN_COMMON_SOURCES
  lsan_common.cc
  lsan_common_linux.cc
  lsan_common_mac.cc)

set(LSAN_SOURCES
  lsan.cc
  lsan_allocator.cc
  lsan_linux.cc
  lsan_interceptors.cc
  lsan_malloc_mac.cc
  lsan_preinit.cc
  lsan_thread.cc)

set(LSAN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_compiler_rt_object_libraries(RTLSanCommon
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${LSAN_COMMON_SUPPORTED_ARCH}
    SOURCES ${LSAN_COMMON_SOURCES}
    CFLAGS ${LSAN_CFLAGS})

if(COMPILER_RT_HAS_LSAN)
  add_compiler_rt_component(lsan)
  if(APPLE)
    add_weak_symbols("lsan" WEAK_SYMBOL_LINK_FLAGS)
    add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

    add_compiler_rt_runtime(clang_rt.lsan
      SHARED
      OS ${SANITIZER_COMMON_SUPPORTED_OS}
      ARCHS ${LSAN_SUPPORTED_ARCH}
      SOURCES ${LSAN_SOURCES}
      OBJECT_LIBS RTLSanCommon
                  RTInterception
                  RTSanitizerCommon
                  RTSanitizerCommonLibc
      CFLAGS ${LSAN_CFLAGS}
      LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS}
      PARENT_TARGET lsan)
  else()
    foreach(arch ${LSAN_SUPPORTED_ARCH})
      add_compiler_rt_runtime(clang_rt.lsan
        STATIC
        ARCHS ${arch}
        SOURCES ${LSAN_SOURCES}
                $<TARGET_OBJECTS:RTInterception.${arch}>
                $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
                $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
                $<TARGET_OBJECTS:RTLSanCommon.${arch}>
        CFLAGS ${LSAN_CFLAGS}
        PARENT_TARGET lsan)
    endforeach()
  endif()
endif()
