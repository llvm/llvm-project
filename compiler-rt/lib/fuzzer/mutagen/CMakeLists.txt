set(MUTAGEN_SOURCES
  Mutagen.cpp
  MutagenCrossOver.cpp
  MutagenDispatcher.cpp
  MutagenUtilPosix.cpp
  MutagenUtilWindows.cpp)

set(MUTAGEN_HEADERS
  Mutagen.h
  MutagenDictionary.h
  MutagenDispatcher.h
  MutagenUtil.h)

# Expose the files in this library to libFuzzer for optimized, direct inclusion.
set(LIBFUZZER_MUTAGEN_SOURCES ${MUTAGEN_SOURCES} PARENT_SCOPE)
set(LIBFUZZER_MUTAGEN_HEADERS ${MUTAGEN_HEADERS} PARENT_SCOPE)

# Reuse the following variables from libFuzzer:
#   FUZZER_SUPPORTED_ARCH
#   FUZZER_SUPPORTED_OS
#   LIBFUZZER_CFLAGS
#   LIBFUZZER_DEPS
include_directories(..)

add_compiler_rt_component(mutagen)

add_compiler_rt_object_libraries(RTmutagen
  OS ${FUZZER_SUPPORTED_OS}
  ARCHS ${FUZZER_SUPPORTED_ARCH}
  SOURCES ${MUTAGEN_SOURCES}
  ADDITIONAL_HEADERS ${MUTAGEN_HEADERS}
  CFLAGS ${LIBFUZZER_CFLAGS}
  DEPS ${LIBFUZZER_DEPS})

add_compiler_rt_runtime(clang_rt.mutagen
  STATIC
  OS ${FUZZER_SUPPORTED_OS}
  ARCHS ${FUZZER_SUPPORTED_ARCH}
  OBJECT_LIBS RTmutagen
  CFLAGS ${LIBFUZZER_CFLAGS}
  PARENT_TARGET mutagen)

if(OS_NAME MATCHES "Linux|Fuchsia" AND
   COMPILER_RT_LIBCXX_PATH AND
   COMPILER_RT_LIBCXXABI_PATH)
  foreach(arch ${FUZZER_SUPPORTED_ARCH})
    get_target_flags_for_arch(${arch} TARGET_CFLAGS)
    set(LIBCXX_${arch}_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libcxx_mutagen_${arch})
    add_custom_libcxx(libcxx_mutagen_${arch} ${LIBCXX_${arch}_PREFIX}
      CFLAGS ${TARGET_CFLAGS}
      CMAKE_ARGS -DCMAKE_CXX_COMPILER_WORKS=ON
                 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                 -DLIBCXXABI_ENABLE_EXCEPTIONS=OFF
                 -DLIBCXX_ABI_NAMESPACE=__Fuzzer)
    target_compile_options(RTmutagen.${arch} PRIVATE -isystem ${LIBCXX_${arch}_PREFIX}/include/c++/v1)
    add_dependencies(RTmutagen.${arch} libcxx_mutagen_${arch}-build)
    partially_link_libcxx(mutagen ${LIBCXX_${arch}_PREFIX} ${arch})
  endforeach()
endif()
