# Build for the ASAN Stable ABI runtime support library.
set(ASAN_ABI_SOURCES
  asan_abi_shim.cpp
  )

set(ASAN_ABI_HEADERS
  ../asan/asan_interface_internal.h
  asan_abi.h
  )

include_directories(..)

add_compiler_rt_component(asan_abi)

if (APPLE)
  # TODO: set in config-ix.cmake
  set(ASAN_ABI_SUPPORTED_OS osx)
  set(ASAN_ABI_SUPPORTED_ARCHS ${X86_64} ${ARM64})
  # Compile Stable API sources into an object library.
  add_compiler_rt_object_libraries(RTASAN_ABI
    OS ${ASAN_ABI_SUPPORTED_OS}
    ARCHS ${ASAN_ABI_SUPPORTED_ARCHS}
    SOURCES ${ASAN_ABI_SOURCES}
    ADDITIONAL_HEADERS ${ASAN_ABI_HEADERS}
    CFLAGS ${SANITIZER_COMMON_CFLAGS})

  add_compiler_rt_runtime(clang_rt.asan_abi
    STATIC
    OS ${ASAN_ABI_SUPPORTED_OS}
    ARCHS ${ASAN_ABI_SUPPORTED_ARCHS}
    OBJECT_LIBS RTASAN_ABI
    CFLAGS ${SANITIZER_COMMON_CFLAGS}
    LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS}
    PARENT_TARGET asan_abi)
endif()
