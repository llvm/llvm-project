# Build for the Cilksan runtime support library.

set(CILKSAN_SOURCES
  cilksan.cpp
  debug_util.cpp
  mem_access.cpp
  print_addr.cpp
  shadow_mem.cpp
  compresseddict_shadow_mem.cpp
  static_dictionary.cpp
  drivercsan.cpp
  csanrt.cpp)

include_directories(${COMPILER_RT_SOURCE_DIR}/include)

set(CILKSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})

append_list_if(COMPILER_RT_HAS_STD_CXX11_FLAG -std=c++11 CILKSAN_CFLAGS)

append_rtti_flag(OFF CILKSAN_CFLAGS)

set(CILKSAN_DYNAMIC_LINK_FLAGS)

set(CILKSAN_DYNAMIC_DEFINITIONS
  ${CILKSAN_COMMON_DEFINITIONS})

set(CILKSAN_DYNAMIC_CFLAGS ${CILKSAN_CFLAGS})

#append_list_if(COMPILER_RT_HAS_LIBC c CILKSAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBDL dl CILKSAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_SNAPPY snappy CILKSAN_DYNAMIC_LIBS)
#append_list_if(COMPILER_RT_HAS_LIBRT rt CILKSAN_DYNAMIC_LIBS)
#append_list_if(COMPILER_RT_HAS_LIBM m CILKSAN_DYNAMIC_LIBS)
#append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread CILKSAN_DYNAMIC_LIBS)
#append_list_if(COMPILER_RT_HAS_LIBSTDCXX stdc++ CILKSAN_DYNAMIC_LIBS)
#append_list_if(COMPILER_RT_HAS_LIBLOG log CILKSAN_DYNAMIC_LIBS)

# Compile CilkSan sources into an object library

add_compiler_rt_object_libraries(RTCilksan_dynamic
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${CILKSAN_SUPPORTED_ARCH}
  SOURCES ${CILKSAN_SOURCES}
  CFLAGS ${CILKSAN_DYNAMIC_CFLAGS}
  DEFS ${CILKSAN_DYNAMIC_DEFINITIONS})

# Build Cilksan runtimes shipped with Clang.
add_compiler_rt_component(cilksan)

foreach (arch ${CILKSAN_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.cilksan
    SHARED
    ARCHS ${arch}
    OBJECT_LIBS RTCilksan_dynamic
    CFLAGS ${CILKSAN_DYNAMIC_CFLAGS}
    LINK_FLAGS ${CILKSAN_DYNAMIC_LINK_FLAGS}
    LINK_LIBS ${CILKSAN_DYNAMIC_LIBS}
    DEFS ${CILKSAN_DYNAMIC_DEFINITIONS}
    PARENT_TARGET cilksan)

  # add_dependencies(cilksan
  #   clang_rt.cilksan-${arch})
endforeach()

if (COMPILER_RT_INCLUDE_TESTS)
  # TODO(bruening): add tests via add_subdirectory(tests)
endif()
