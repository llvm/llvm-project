include_directories(..)

set(LOWFAT_COMMON_DEFINITIONS)

#===------------------------------------------------------------------------===
# Custom size-class configuration (non-POW2 support)
#
# Set -DLOWFAT_SIZES_CFG=/path/to/sizes.cfg at cmake time to enable.
# This builds the lf_config_gen host tool, runs it to produce
# lf_config_generated.h, and compiles the runtime with LOWFAT_CUSTOM_CONFIG.
#===------------------------------------------------------------------------===
set(LOWFAT_SIZES_CFG "" CACHE FILEPATH
  "Path to sizes.cfg for non-POW2 LowFat size class generation. Empty = POW2-only mode.")

if(LOWFAT_SIZES_CFG)
  # ---- Build the generator tool (host compiler, no LLVM deps) ----
  # Use add_custom_target / add_custom_command rather than add_executable so
  # this host tool is never cross-compiled.
  set(LOWFAT_CONFIG_GEN_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/lf_config_gen.c)
  set(LOWFAT_CONFIG_GEN_BIN
      ${CMAKE_CURRENT_BINARY_DIR}/lf_config_gen${CMAKE_EXECUTABLE_SUFFIX})
  set(LOWFAT_GENERATED_HEADER
      ${CMAKE_CURRENT_BINARY_DIR}/lf_config_generated.h)

  add_custom_command(
    OUTPUT  ${LOWFAT_CONFIG_GEN_BIN}
    COMMAND ${CMAKE_C_COMPILER} -std=c11 -O2
            ${LOWFAT_CONFIG_GEN_SRC}
            -o ${LOWFAT_CONFIG_GEN_BIN}
    DEPENDS ${LOWFAT_CONFIG_GEN_SRC}
    COMMENT "Compiling lf_config_gen host tool"
    VERBATIM
  )

  add_custom_command(
    OUTPUT  ${LOWFAT_GENERATED_HEADER}
    COMMAND ${LOWFAT_CONFIG_GEN_BIN} ${LOWFAT_SIZES_CFG} ${LOWFAT_GENERATED_HEADER}
    DEPENDS ${LOWFAT_CONFIG_GEN_BIN} ${LOWFAT_SIZES_CFG}
    COMMENT "Generating LowFat size tables from ${LOWFAT_SIZES_CFG}"
    VERBATIM
  )

  add_custom_target(lowfat_config_generated DEPENDS ${LOWFAT_GENERATED_HEADER})
  set(LOWFAT_DEPS lowfat_config_generated)

  # Make the generated header visible to all sources in this directory.
  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  # Activate the custom-config code paths in the runtime AND the LLVM pass.
  list(APPEND LOWFAT_COMMON_DEFINITIONS LOWFAT_CUSTOM_CONFIG=1)

  message(STATUS "LowFat: custom config enabled: ${LOWFAT_SIZES_CFG}")
  message(STATUS "LowFat: generated header:      ${LOWFAT_GENERATED_HEADER}")
else()
  message(STATUS "LowFat: using default POW2-only mode (no LOWFAT_SIZES_CFG set)")
  set(LOWFAT_DEPS)
endif()
set(LOWFAT_SOURCES
  lf_rtl.cpp
  lf_interceptors.cpp
)

set(LOWFAT_HEADERS
  lf_allocator.h
  lf_config.h
  lf_interface.h
)

set(LOWFAT_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF LOWFAT_CFLAGS)

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
      OBJECT_LIBS RTInterception
                  RTSanitizerCommon
                  RTSanitizerCommonLibc
                  RTSanitizerCommonSymbolizer
      CFLAGS ${LOWFAT_CFLAGS}
      LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS}
      DEFS ${LOWFAT_COMMON_DEFINITIONS}
      DEPS ${LOWFAT_DEPS}
      PARENT_TARGET lowfat)
  else()
    add_compiler_rt_runtime(clang_rt.lowfat
      STATIC
      ARCHS ${LOWFAT_SUPPORTED_ARCH}
      SOURCES ${LOWFAT_SOURCES}
      ADDITIONAL_HEADERS ${LOWFAT_HEADERS}
      OBJECT_LIBS RTInterception
                  RTSanitizerCommon
                  RTSanitizerCommonLibc
                  RTSanitizerCommonSymbolizer
                  RTSanitizerCommonSymbolizerInternal
      CFLAGS ${LOWFAT_CFLAGS}
      DEFS ${LOWFAT_COMMON_DEFINITIONS}
      DEPS ${LOWFAT_DEPS}
      PARENT_TARGET lowfat)
  endif()
endif()
