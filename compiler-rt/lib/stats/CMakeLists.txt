include_directories(..)

add_custom_target(stats)
set_target_properties(stats PROPERTIES FOLDER "Compiler-RT Misc")

if(APPLE)
  set(STATS_LIB_FLAVOR SHARED)

  add_weak_symbols("asan" WEAK_SYMBOL_LINK_FLAGS)
  add_weak_symbols("ubsan" WEAK_SYMBOL_LINK_FLAGS)
  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)
else()
  set(STATS_LIB_FLAVOR STATIC)

  set(WEAK_SYMBOL_LINK_FLAGS)
endif()

add_compiler_rt_runtime(clang_rt.stats
  ${STATS_LIB_FLAVOR}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  SOURCES stats.cc
  OBJECT_LIBS RTSanitizerCommon
              RTSanitizerCommonLibc
  CFLAGS ${SANITIZER_COMMON_CFLAGS}
  LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS}
  PARENT_TARGET stats)

add_compiler_rt_runtime(clang_rt.stats_client
  STATIC
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  SOURCES stats_client.cc
  CFLAGS ${SANITIZER_COMMON_CFLAGS}
  LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS}
  PARENT_TARGET stats)
