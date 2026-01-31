include_directories(..)

set(LOWFAT_SOURCES
  lf_rtl.cpp
)

set(LOWFAT_HEADERS
  lf_interface.h
)

set(LOWFAT_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF LOWFAT_CFLAGS)
set(LOWFAT_COMMON_DEFINITIONS)

# Static runtime library.
add_compiler_rt_component(lowfat)

if(COMPILER_RT_HAS_LOWFAT)
  if(APPLE)
    add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

    add_compiler_rt_runtime(clang_rt.lowfat
      SHARED
      OS ${SANITIZER_COMMON_SUPPORTED_OS}
      ARCHS ${LOWFAT_SUPPORTED_ARCH}
      SOURCES ${LOWFAT_SOURCES}
      ADDITIONAL_HEADERS ${LOWFAT_HEADERS}
      OBJECT_LIBS RTSanitizerCommon
                  RTSanitizerCommonLibc
                  RTSanitizerCommonSymbolizer
      CFLAGS ${LOWFAT_CFLAGS}
      LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS}
      DEFS ${LOWFAT_COMMON_DEFINITIONS}
      PARENT_TARGET lowfat)
  else()
    add_compiler_rt_runtime(clang_rt.lowfat
      STATIC
      ARCHS ${LOWFAT_SUPPORTED_ARCH}
      SOURCES ${LOWFAT_SOURCES}
      ADDITIONAL_HEADERS ${LOWFAT_HEADERS}
      OBJECT_LIBS RTSanitizerCommon
                  RTSanitizerCommonLibc
                  RTSanitizerCommonSymbolizer
                  RTSanitizerCommonSymbolizerInternal
      CFLAGS ${LOWFAT_CFLAGS}
      DEFS ${LOWFAT_COMMON_DEFINITIONS}
      PARENT_TARGET lowfat)
  endif()
endif()
