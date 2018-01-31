# Build for the Cilkscale runtime support library.

set(CILKSCALE_SOURCES
  cilkscale.c
  csanrt.c)

include_directories(..)

set(CILKSCALE_CFLAGS ${SANITIZER_COMMON_CFLAGS})

append_list_if(COMPILER_RT_HAS_STD_C11_FLAG -std=c11 CILKSCALE_CFLAGS)

append_rtti_flag(OFF CILKSCALE_CFLAGS)

# Build Cilkscale runtimes shipped with Clang.
add_compiler_rt_component(cilkscale)

foreach (arch ${CILKSCALE_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.cilkscale
    STATIC
    ARCHS ${arch}
    SOURCES ${CILKSCALE_SOURCES}
    CFLAGS ${CILKSCALE_CFLAGS}
    PARENT_TARGET cilkscale)
endforeach()

if (COMPILER_RT_INCLUDE_TESTS)
  # TODO(bruening): add tests via add_subdirectory(tests)
endif()
