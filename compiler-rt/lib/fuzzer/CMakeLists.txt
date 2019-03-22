set(LIBFUZZER_SOURCES
  FuzzerCrossOver.cpp
  FuzzerDataFlowTrace.cpp
  FuzzerDriver.cpp
  FuzzerExtFunctionsDlsym.cpp
  FuzzerExtFunctionsWeak.cpp
  FuzzerExtFunctionsWindows.cpp
  FuzzerExtraCounters.cpp
  FuzzerFork.cpp
  FuzzerIO.cpp
  FuzzerIOPosix.cpp
  FuzzerIOWindows.cpp
  FuzzerLoop.cpp
  FuzzerMerge.cpp
  FuzzerMutate.cpp
  FuzzerSHA1.cpp
  FuzzerTracePC.cpp
  FuzzerUtil.cpp
  FuzzerUtilDarwin.cpp
  FuzzerUtilFuchsia.cpp
  FuzzerUtilLinux.cpp
  FuzzerUtilPosix.cpp
  FuzzerUtilWindows.cpp)

set(LIBFUZZER_HEADERS
  FuzzerBuiltins.h
  FuzzerBuiltinsMsvc.h
  FuzzerCommand.h
  FuzzerCorpus.h
  FuzzerDataFlowTrace.h
  FuzzerDefs.h
  FuzzerDictionary.h
  FuzzerExtFunctions.def
  FuzzerExtFunctions.h
  FuzzerFlags.def
  FuzzerFork.h
  FuzzerIO.h
  FuzzerInterface.h
  FuzzerInternal.h
  FuzzerMerge.h
  FuzzerMutate.h
  FuzzerOptions.h
  FuzzerRandom.h
  FuzzerSHA1.h
  FuzzerTracePC.h
  FuzzerUtil.h
  FuzzerValueBitMap.h)

CHECK_CXX_SOURCE_COMPILES("
  static thread_local int blah;
  int main() {
  return 0;
  }
  " HAS_THREAD_LOCAL)

set(LIBFUZZER_CFLAGS ${SANITIZER_COMMON_CFLAGS})

if(OS_NAME MATCHES "Linux|Fuchsia" AND
   COMPILER_RT_LIBCXX_PATH AND
   COMPILER_RT_LIBCXXABI_PATH)
  list(APPEND LIBFUZZER_CFLAGS -nostdinc++ -D_LIBCPP_ABI_VERSION=Fuzzer)
  # Remove -stdlib= which is unused when passing -nostdinc++.
  string(REGEX REPLACE "-stdlib=[a-zA-Z+]*" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
elseif(TARGET cxx-headers OR HAVE_LIBCXX)
  set(LIBFUZZER_DEPS cxx-headers)
endif()

append_list_if(COMPILER_RT_HAS_OMIT_FRAME_POINTER_FLAG -fno-omit-frame-pointer LIBFUZZER_CFLAGS)

if (CMAKE_CXX_FLAGS MATCHES "fsanitize-coverage")
  list(APPEND LIBFUZZER_CFLAGS -fno-sanitize-coverage=trace-pc-guard,edge,trace-cmp,indirect-calls,8bit-counters)
endif()

if(MSVC)
  # Silence warnings by turning off exceptions in MSVC headers and avoid an
  # error by unecessarily defining thread_local when it isn't even used on
  # Windows.
  list(APPEND LIBFUZZER_CFLAGS -D_HAS_EXCEPTIONS=0)
else()
  if(NOT HAS_THREAD_LOCAL)
    list(APPEND LIBFUZZER_CFLAGS -Dthread_local=__thread)
  endif()
endif()

set(FUZZER_SUPPORTED_OS ${SANITIZER_COMMON_SUPPORTED_OS})

add_compiler_rt_object_libraries(RTfuzzer
  OS ${FUZZER_SUPPORTED_OS}
  ARCHS ${FUZZER_SUPPORTED_ARCH}
  SOURCES ${LIBFUZZER_SOURCES}
  ADDITIONAL_HEADERS ${LIBFUZZER_HEADERS}
  CFLAGS ${LIBFUZZER_CFLAGS}
  DEPS ${LIBFUZZER_DEPS})

add_compiler_rt_object_libraries(RTfuzzer_main
  OS ${FUZZER_SUPPORTED_OS}
  ARCHS ${FUZZER_SUPPORTED_ARCH}
  SOURCES FuzzerMain.cpp
  CFLAGS ${LIBFUZZER_CFLAGS}
  DEPS ${LIBFUZZER_DEPS})

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

if(OS_NAME MATCHES "Linux|Fuchsia" AND
   COMPILER_RT_LIBCXX_PATH AND
   COMPILER_RT_LIBCXXABI_PATH)
  macro(partially_link_libcxx name dir arch)
    set(cxx_${arch}_merge_dir "${CMAKE_CURRENT_BINARY_DIR}/cxx_${arch}_merge.dir")
    file(MAKE_DIRECTORY ${cxx_${arch}_merge_dir})
    add_custom_command(TARGET clang_rt.${name}-${arch} POST_BUILD
      COMMAND ${CMAKE_LINKER} --whole-archive "$<TARGET_LINKER_FILE:clang_rt.${name}-${arch}>" --no-whole-archive ${dir}/lib/libc++.a -r -o ${name}.o
      COMMAND ${CMAKE_OBJCOPY} --localize-hidden ${name}.o
      COMMAND ${CMAKE_COMMAND} -E remove "$<TARGET_LINKER_FILE:clang_rt.${name}-${arch}>"
      COMMAND ${CMAKE_AR} qcs "$<TARGET_LINKER_FILE:clang_rt.${name}-${arch}>" ${name}.o
      WORKING_DIRECTORY ${cxx_${arch}_merge_dir}
    )
  endmacro()

  foreach(arch ${FUZZER_SUPPORTED_ARCH})
    get_target_flags_for_arch(${arch} TARGET_CFLAGS)
    set(LIBCXX_${arch}_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libcxx_fuzzer_${arch})
    add_custom_libcxx(libcxx_fuzzer_${arch} ${LIBCXX_${arch}_PREFIX}
      CFLAGS ${TARGET_CFLAGS}
      CMAKE_ARGS -DCMAKE_CXX_COMPILER_WORKS=ON
                 -DLIBCXX_ABI_NAMESPACE=Fuzzer)
    target_compile_options(RTfuzzer.${arch} PRIVATE -isystem ${LIBCXX_${arch}_PREFIX}/include/c++/v1)
    add_dependencies(RTfuzzer.${arch} libcxx_fuzzer_${arch}-build)
    target_compile_options(RTfuzzer_main.${arch} PRIVATE -isystem ${LIBCXX_${arch}_PREFIX}/include/c++/v1)
    add_dependencies(RTfuzzer_main.${arch} libcxx_fuzzer_${arch}-build)
    partially_link_libcxx(fuzzer_no_main ${LIBCXX_${arch}_PREFIX} ${arch})
    partially_link_libcxx(fuzzer ${LIBCXX_${arch}_PREFIX} ${arch})
  endforeach()
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
