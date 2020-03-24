# Build for the ComprehensiveStaticInstrumentation runtime support library.

add_custom_target(csi)

set(CSI_RTL_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_list_if(COMPILER_RT_HAS_STD_C11_FLAG -std=c11 CSI_RTL_CFLAGS)
append_rtti_flag(OFF CSI_RTL_CFLAGS)

include_directories(${COMPILER_RT_SOURCE_DIR}/include)

set(CSI_SOURCES csirt.c)

foreach (arch ${CSI_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.csi
    SHARED
    ARCHS ${arch}
    SOURCES ${CSI_SOURCES}
    CFLAGS ${CSI_RTL_CFLAGS}
    PARENT_TARGET csi)
  add_compiler_rt_runtime(clang_rt.csi
    STATIC
    ARCHS ${arch}
    SOURCES ${CSI_SOURCES}
    CFLAGS ${CSI_RTL_CFLAGS}
    PARENT_TARGET csi)
endforeach()

add_dependencies(compiler-rt csi)

if (COMPILER_RT_INCLUDE_TESTS)
  # TODO(bruening): add tests via add_subdirectory(tests)
endif()
