set(LIBFUZZER_SOURCES
  FuzzerClangCounters.cpp
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

CHECK_CXX_SOURCE_COMPILES("
  static thread_local int blah;
  int main() {
  return 0;
  }
  " HAS_THREAD_LOCAL)

set(LIBFUZZER_CFLAGS ${SANITIZER_COMMON_CFLAGS})

append_list_if(COMPILER_RT_HAS_OMIT_FRAME_POINTER_FLAG -fno-omit-frame-pointer LIBFUZZER_CFLAGS)

if (CMAKE_CXX_FLAGS MATCHES "fsanitize-coverage")
  list(APPEND LIBFUZZER_CFLAGS -fno-sanitize-coverage=trace-pc-guard,edge,trace-cmp,indirect-calls,8bit-counters)
endif()

if(NOT HAS_THREAD_LOCAL)
  list(APPEND LIBFUZZER_CFLAGS -Dthread_local=__thread)
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
