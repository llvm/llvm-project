# Build for the DeadlockSanitizer runtime support library.

include_directories(..)

set(TSAN_DEADLOCK_SOURCES
  tsan_deadlock_rtl.cpp
  tsan_deadlock_interceptors.cpp
  tsan_deadlock_interface.cpp)

set(TSAN_DEADLOCK_HEADERS
  tsan_deadlock_rtl.h)

set(TSAN_DEADLOCK_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF TSAN_DEADLOCK_CFLAGS)

set(TSAN_DEADLOCK_DYNAMIC_LINK_LIBS
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_CXX_ABI_LIBRARIES}
  ${SANITIZER_COMMON_LINK_LIBS})

append_list_if(COMPILER_RT_HAS_LIBDL dl TSAN_DEADLOCK_DYNAMIC_LINK_LIBS)
append_list_if(COMPILER_RT_HAS_LIBM m TSAN_DEADLOCK_DYNAMIC_LINK_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread TSAN_DEADLOCK_DYNAMIC_LINK_LIBS)

add_compiler_rt_component(tsan_deadlock)

if(APPLE)
  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

  add_compiler_rt_runtime(clang_rt.tsan_deadlock
    SHARED
    OS ${TSAN_DEADLOCK_SUPPORTED_OS}
    ARCHS ${TSAN_DEADLOCK_SUPPORTED_ARCH}
    SOURCES ${TSAN_DEADLOCK_SOURCES}
    ADDITIONAL_HEADERS ${TSAN_DEADLOCK_HEADERS}
    OBJECT_LIBS RTInterception
                RTSanitizerCommon
                RTSanitizerCommonLibc
                RTSanitizerCommonCoverage
                RTSanitizerCommonSymbolizer
    CFLAGS ${TSAN_DEADLOCK_CFLAGS}
    LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
    LINK_LIBS ${SANITIZER_COMMON_LINK_LIBS}
    PARENT_TARGET tsan_deadlock)
else()
  foreach(arch ${TSAN_DEADLOCK_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.tsan_deadlock
      STATIC
      ARCHS ${arch}
      SOURCES ${TSAN_DEADLOCK_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonSymbolizerInternal.${arch}>
      ADDITIONAL_HEADERS ${TSAN_DEADLOCK_HEADERS}
      CFLAGS ${TSAN_DEADLOCK_CFLAGS}
      PARENT_TARGET tsan_deadlock)
    add_compiler_rt_runtime(clang_rt.tsan_deadlock
      SHARED
      ARCHS ${arch}
      SOURCES ${TSAN_DEADLOCK_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonSymbolizerInternal.${arch}>
      ADDITIONAL_HEADERS ${TSAN_DEADLOCK_HEADERS}
      CFLAGS ${TSAN_DEADLOCK_CFLAGS}
      LINK_LIBS ${TSAN_DEADLOCK_DYNAMIC_LINK_LIBS}
      LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS}
      PARENT_TARGET tsan_deadlock)
  endforeach()
endif()
