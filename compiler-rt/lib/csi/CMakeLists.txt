# Build for the ComprehensiveStaticInstrumentation runtime support library.

add_custom_target(csi)

set(CSI_RTL_CFLAGS ${SANITIZER_COMMON_CFLAGS} -std=c11)
append_rtti_flag(OFF CSI_RTL_CFLAGS)

include_directories(..)

set(CSI_SOURCES csirt.c)

foreach (arch ${CSI_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.csi
    STATIC
    ARCHS ${arch}
    SOURCES ${CSI_SOURCES}
    CFLAGS ${CSI_RTL_CFLAGS})
  add_dependencies(csi
    clang_rt.csi-${arch})
endforeach()

add_dependencies(compiler-rt csi)

if (COMPILER_RT_INCLUDE_TESTS)
  # TODO(bruening): add tests via add_subdirectory(tests)
endif()
