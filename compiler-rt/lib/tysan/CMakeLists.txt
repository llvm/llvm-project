include_directories(..)

# Runtime library sources and build flags.
set(TYSAN_SOURCES
  tysan.cpp
  tysan_interceptors.cpp
  )

SET(TYSAN_HEADERS
  tysan.h
  tysan_flags.inc
  tysan_platform.h
  )

set(TYSAN_COMMON_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF TYSAN_COMMON_CFLAGS)
# Prevent clang from generating libc calls.
append_list_if(COMPILER_RT_HAS_FFREESTANDING_FLAG -ffreestanding TYSAN_COMMON_CFLAGS)
set(TYSAN_DYNAMIC_CFLAGS ${TYSAN_COMMON_CFLAGS})

set(TYSAN_COMMON_DEFINITIONS "")
set(TYSAN_DYNAMIC_DEFINITIONS ${TYSAN_COMMON_DEFINITIONS} TYSAN_DYNAMIC=1)

# Compile TYSan sources into an object library.

add_compiler_rt_object_libraries(RTTysan_dynamic
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${TYSAN_SUPPORTED_ARCH}
  SOURCES ${TYSAN_SOURCES}
  ADDITIONAL_HEADERS ${TYSAN_HEADERS}
  CFLAGS ${TYSAN_DYNAMIC_CFLAGS}
  DEFS ${TYSAN_DYNAMIC_DEFINITIONS})


# Static runtime library.
add_compiler_rt_component(tysan)


if(APPLE)
  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

  add_compiler_rt_runtime(clang_rt.tysan
    SHARED
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${TYSAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTTysan_dynamic
                RTInterception
                RTSanitizerCommon
                RTSanitizerCommonLibc
                RTSanitizerCommonSymbolizer
    CFLAGS ${TYSAN_DYNAMIC_CFLAGS}
    LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS}
    DEFS ${TYSAN_DYNAMIC_DEFINITIONS}
    PARENT_TARGET tysan)

  add_compiler_rt_runtime(clang_rt.tysan_static
    STATIC
    ARCHS ${TYSAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTTysan_static
    CFLAGS ${TYSAN_CFLAGS}
    DEFS ${TYSAN_COMMON_DEFINITIONS}
    PARENT_TARGET tysan)
else()
  set(TYSAN_CFLAGS ${TYSAN_COMMON_CFLAGS})
  append_list_if(COMPILER_RT_HAS_FPIE_FLAG -fPIE TYSAN_CFLAGS)

  foreach(arch ${TYSAN_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.tysan
      STATIC
      ARCHS ${arch}
      SOURCES ${TYSAN_SOURCES}
      OBJECT_LIBS RTInterception
                  RTSanitizerCommon
                  RTSanitizerCommonLibc
                  RTSanitizerCommonSymbolizer
      CFLAGS ${TYSAN_CFLAGS}
      PARENT_TARGET tysan)
  endforeach()
endif()
