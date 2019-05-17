add_compiler_rt_component(gwp_asan)

include_directories(..)

set(GWP_ASAN_SOURCES
  random.cpp
)

set(GWP_ASAN_HEADERS
  random.h
)

# Ensure that GWP-ASan meets the delegated requirements of some supporting
# allocators. Some supporting allocators (e.g. scudo standalone) cannot use any
# parts of the C++ standard library.
set(GWP_ASAN_CFLAGS -fno-rtti -fno-exceptions -nostdinc++)

if (COMPILER_RT_HAS_GWP_ASAN)
  foreach(arch ${GWP_ASAN_SUPPORTED_ARCH})
    add_compiler_rt_runtime(
      clang_rt.gwp_asan
      STATIC
      ARCHS ${arch}
      SOURCES ${GWP_ASAN_SOURCES}
      ADDITIONAL_HEADERS ${GWP_ASAN_HEADERS}
      CFLAGS ${GWP_ASAN_CFLAGS}
      PARENT_TARGET gwp_asan
    )
  endforeach()

  add_compiler_rt_object_libraries(RTGwpAsan
      ARCHS ${GWP_ASAN_SUPPORTED_ARCH}
      SOURCES ${GWP_ASAN_SOURCES}
      ADDITIONAL_HEADERS ${GWP_ASAN_HEADERS}
      CFLAGS ${GWP_ASAN_CFLAGS})
endif()
