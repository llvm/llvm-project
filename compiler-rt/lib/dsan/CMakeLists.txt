include_directories(..)

set(DSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF DSAN_CFLAGS)

# Too many existing bugs, needs cleanup.
append_list_if(COMPILER_RT_HAS_WNO_FORMAT -Wno-format DSAN_CFLAGS)

set(DSAN_COMMON_SOURCES
  dsan_common.cpp
  )

set(DSAN_SOURCES
  dsan.cpp
  dsan_allocator.cpp
  dsan_fuchsia.cpp
  dsan_interceptors.cpp
  dsan_linux.cpp
  dsan_mac.cpp
  dsan_malloc_mac.cpp
  dsan_posix.cpp
  dsan_preinit.cpp
  dsan_thread.cpp
  )

set(DSAN_HEADERS
  dsan.h
  dsan_allocator.h
  dsan_common.h
  dsan_thread.h
  )

set(DSAN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Shared DSan runtime functionality.
add_compiler_rt_object_libraries(RTDSanCommon
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${DSAN_COMMON_SUPPORTED_ARCH}
    SOURCES ${DSAN_COMMON_SOURCES}
    ADDITIONAL_HEADERS ${DSAN_HEADERS}
    CFLAGS ${DSAN_CFLAGS})

if(COMPILER_RT_HAS_DSAN)
  add_compiler_rt_component(dsan)
  if(APPLE)
    set(DSAN_LINK_LIBS ${SANITIZER_COMMON_LINK_LIBS})

    add_weak_symbols("dsan" WEAK_SYMBOL_LINK_FLAGS)
    add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

    add_compiler_rt_runtime(clang_rt.dsan
      SHARED
      OS ${DSAN_SUPPORTED_OS}
      ARCHS ${DSAN_SUPPORTED_ARCH}
      SOURCES ${DSAN_SOURCES}
      ADDITIONAL_HEADERS ${DSAN_HEADERS}
      OBJECT_LIBS RTDSanCommon
                  RTInterception
                  RTSanitizerCommon
                  RTSanitizerCommonLibc
                  RTSanitizerCommonCoverage
                  RTSanitizerCommonSymbolizer
      CFLAGS ${DSAN_CFLAGS}
      LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
      LINK_LIBS ${DSAN_LINK_LIBS}
      PARENT_TARGET dsan)
  else()
    foreach(arch ${DSAN_SUPPORTED_ARCH})
      add_compiler_rt_runtime(clang_rt.dsan
        STATIC
        ARCHS ${arch}
        SOURCES ${DSAN_SOURCES}
                $<TARGET_OBJECTS:RTInterception.${arch}>
                $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
                $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
                $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
                $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
                $<TARGET_OBJECTS:RTSanitizerCommonSymbolizerInternal.${arch}>
                $<TARGET_OBJECTS:RTDSanCommon.${arch}>
        ADDITIONAL_HEADERS ${DSAN_HEADERS}
        CFLAGS ${DSAN_CFLAGS}
        PARENT_TARGET dsan)
    endforeach()
  endif()
endif()
