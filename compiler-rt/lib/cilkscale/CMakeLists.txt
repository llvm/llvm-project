# Build for the Cilkscale runtime support library.

set(CILKSCALE_SOURCES
  cilkscale.c
  csanrt.c)

include_directories(${COMPILER_RT_SOURCE_DIR}/include)

set(CILKSCALE_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_list_if(COMPILER_RT_HAS_STD_C11_FLAG -std=c11 CILKSCALE_CFLAGS)
append_rtti_flag(OFF CILKSCALE_CFLAGS)

set(CILKSCALE_DYNAMIC_LINK_FLAGS)

set(CILKSCALE_DYNAMIC_CFLAGS ${CILKSCALE_CFLAGS})

set(CILKSCALE_COMMON_LIBS ${SANITIZER_CXX_ABI_LIBRARY} ${SANITIZER_COMMON_LINK_LIBS})

set(CILKSCALE_DYNAMIC_LIBS ${CILKSCALE_COMMON_LIBS})

# Build Cilkscale runtimes shipped with Clang.
add_compiler_rt_component(cilkscale)

foreach (arch ${CILKSCALE_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.cilkscale
    STATIC
    ARCHS ${arch}
    SOURCES ${CILKSCALE_SOURCES}
    CFLAGS ${CILKSCALE_CFLAGS}
    PARENT_TARGET cilkscale)

  add_compiler_rt_runtime(clang_rt.cilkscale
    SHARED
    ARCHS ${arch}
    SOURCES ${CILKSCALE_SOURCES}
    CFLAGS ${CILKSCALE_DYNAMIC_CFLAGS}
    LINK_FLAGS ${CILKSCALE_DYNAMIC_LINK_FLAGS}
    LINK_LIBS ${CILKSCALE_DYNAMIC_LIBS}
    PARENT_TARGET cilksan)
endforeach()

if (COMPILER_RT_INCLUDE_TESTS)
  # TODO(bruening): add tests via add_subdirectory(tests)
endif()
