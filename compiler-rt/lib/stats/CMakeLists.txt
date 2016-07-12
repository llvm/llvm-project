include_directories(..)

add_custom_target(stats)
set_target_properties(stats PROPERTIES FOLDER "Compiler-RT Misc")

if(APPLE)
  set(STATS_LIB_FLAVOR SHARED)
else()
  set(STATS_LIB_FLAVOR STATIC)
endif()

add_compiler_rt_runtime(clang_rt.stats
  ${STATS_LIB_FLAVOR}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  SOURCES stats.cc
  OBJECT_LIBS RTSanitizerCommon
              RTSanitizerCommonLibc
  CFLAGS ${SANITIZER_COMMON_CFLAGS}
  PARENT_TARGET stats)

add_compiler_rt_runtime(clang_rt.stats_client
  STATIC
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  SOURCES stats_client.cc
  CFLAGS ${SANITIZER_COMMON_CFLAGS}
  PARENT_TARGET stats)
