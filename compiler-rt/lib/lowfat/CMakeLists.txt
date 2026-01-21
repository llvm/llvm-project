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
  foreach(arch ${LOWFAT_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.lowfat
      STATIC
      ARCHS ${arch}
      SOURCES ${LOWFAT_SOURCES}
      ADDITIONAL_HEADERS ${LOWFAT_HEADERS}
      OBJECT_LIBS RTSanitizerCommon
                  RTSanitizerCommonLibc
                  RTSanitizerCommonSymbolizer
      CFLAGS ${LOWFAT_CFLAGS}
      DEFS ${LOWFAT_COMMON_DEFINITIONS}
      PARENT_TARGET lowfat)
  endforeach()
endif()
