set(CSAN_SOURCES
  csan_gpu.cpp
  )

include_directories(..)

include_directories(${COMPILER_RT_SOURCE_DIR}/include)
include(FindLibcCommonUtils)
get_target_property(LIBC_COMMON_UTILS_INCLUDE
  llvm-libc-common-utilities INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${LIBC_COMMON_UTILS_INCLUDE})

set(CSAN_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS)
append_rtti_flag(OFF CSAN_CFLAGS)

add_compiler_rt_component(csan)

add_compiler_rt_object_libraries(RTCsan
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${UBSAN_COMMON_SUPPORTED_ARCH}
  SOURCES ${CSAN_SOURCES} CFLAGS ${CSAN_CFLAGS})

add_compiler_rt_runtime(clang_rt.csan
  STATIC
  OS ${UBSAN_SUPPORTED_OS}
  ARCHS ${UBSAN_SUPPORTED_ARCH}
  OBJECT_LIBS RTCsan
  CFLAGS ${CSAN_CFLAGS}
  PARENT_TARGET csan)
