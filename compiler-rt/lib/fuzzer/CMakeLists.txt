set(LIBFUZZER_SOURCES
  FuzzerCrossOver.cpp
  FuzzerDriver.cpp
  FuzzerExtFunctionsDlsym.cpp
  FuzzerExtFunctionsDlsymWin.cpp
  FuzzerExtFunctionsWeak.cpp
  FuzzerExtraCounters.cpp
  FuzzerIO.cpp
  FuzzerIOPosix.cpp
  FuzzerIOWindows.cpp
  FuzzerLoop.cpp
  FuzzerMerge.cpp
  FuzzerMutate.cpp
  FuzzerSHA1.cpp
  FuzzerShmemPosix.cpp
  FuzzerShmemWindows.cpp
  FuzzerTracePC.cpp
  FuzzerUtil.cpp
  FuzzerUtilDarwin.cpp
  FuzzerUtilLinux.cpp
  FuzzerUtilPosix.cpp
  FuzzerUtilWindows.cpp
  )

if (CMAKE_CXX_FLAGS MATCHES "fsanitize-coverage")
  set(LIBFUZZER_CFLAGS -fno-sanitize-coverage=trace-pc-guard,edge,trace-cmp,indirect-calls,8bit-counters)
endif()

if(APPLE)
  set(FUZZER_SUPPORTED_OS osx)
endif()

add_compiler_rt_object_libraries(RTfuzzer
  OS ${FUZZER_SUPPORTED_OS}
  ARCHS ${FUZZER_SUPPORTED_ARCH}
  SOURCES ${LIBFUZZER_SOURCES}
  CFLAGS ${LIBFUZZER_CFLAGS})

add_compiler_rt_object_libraries(RTfuzzer_main
  OS ${FUZZER_SUPPORTED_OS}
  ARCHS ${FUZZER_SUPPORTED_ARCH}
  SOURCES FuzzerMain.cpp
  CFLAGS ${LIBFUZZER_CFLAGS})

add_compiler_rt_runtime(clang_rt.fuzzer
  STATIC
  OS ${FUZZER_SUPPORTED_OS}
  ARCHS ${FUZZER_SUPPORTED_ARCH}
  OBJECT_LIBS RTfuzzer RTfuzzer_main
  CFLAGS ${LIBFUZZER_CFLAGS}
  PARENT_TARGET fuzzer)

add_compiler_rt_runtime(clang_rt.fuzzer_no_main
  STATIC
  OS ${FUZZER_SUPPORTED_OS}
  ARCHS ${FUZZER_SUPPORTED_ARCH}
  OBJECT_LIBS RTfuzzer
  CFLAGS ${LIBFUZZER_CFLAGS}
  PARENT_TARGET fuzzer)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
