include(CheckCXXCompilerFlag)
include(CompilerRTCompile)
include(CompilerRTLink)

include_directories(..)
include_directories(../..)

set(MSAN_LIBCXX_CFLAGS
  -fsanitize=memory
  -fsanitize-memory-track-origins
  -Wno-pedantic)

# Unittest sources and build flags.
set(MSAN_UNITTEST_SOURCES msan_test.cc msan_test_main.cc)
set(MSAN_LOADABLE_SOURCE msan_loadable.cc)
set(MSAN_UNITTEST_HEADERS
  msan_test_config.h
  ../../../include/sanitizer/msan_interface.h
)
set(MSAN_UNITTEST_COMMON_CFLAGS
  -nostdinc++
  -isystem ${COMPILER_RT_LIBCXX_PATH}/include
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/msan
  -g
  -O2
  -fno-exceptions
  -fno-omit-frame-pointer
  -mno-omit-leaf-frame-pointer
  -Wno-deprecated-declarations
  -Wno-unused-variable
  -Wno-zero-length-array
  -Wno-uninitialized
  -Werror=sign-compare
)
set(MSAN_UNITTEST_INSTRUMENTED_CFLAGS
  ${MSAN_UNITTEST_COMMON_CFLAGS}
  -fsanitize=memory
  -fsanitize-memory-track-origins
  -mllvm -msan-keep-going=1
)
set(MSAN_UNITTEST_LINK_FLAGS
  -fsanitize=memory
  # Don't need -stdlib=libc++ because we explicitly list libc++.so in the linker
  # inputs.
  # FIXME: we build libcxx without cxxabi and need libstdc++ to provide it.
  -lstdc++
)

append_list_if(COMPILER_RT_HAS_LIBDL -ldl MSAN_UNITTEST_LINK_FLAGS)

# Compile source for the given architecture, using compiler
# options in ${ARGN}, and add it to the object list.
macro(msan_compile obj_list source arch kind)
  get_filename_component(basename ${source} NAME)
  set(output_obj "${basename}.${arch}${kind}.o")
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  set(COMPILE_DEPS ${MSAN_UNITTEST_HEADERS})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND COMPILE_DEPS gtest msan)
  endif()
  clang_compile(${output_obj} ${source}
                CFLAGS ${ARGN} ${TARGET_CFLAGS}
                DEPS ${COMPILE_DEPS})
  list(APPEND ${obj_list} ${output_obj})
endmacro()

macro(msan_link_shared so_list so_name arch kind)
  cmake_parse_arguments(SOURCE "" "" "OBJECTS;LINK_FLAGS;DEPS" ${ARGN})
  set(output_so "${CMAKE_CURRENT_BINARY_DIR}/${so_name}.${arch}${kind}.so")
  get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND SOURCE_DEPS msan)
  endif()
  clang_link_shared(${output_so}
                OBJECTS ${SOURCE_OBJECTS}
                LINK_FLAGS ${TARGET_LINK_FLAGS} ${SOURCE_LINK_FLAGS}
                DEPS ${SOURCE_DEPS})
  list(APPEND ${so_list} ${output_so})
endmacro()

# Main MemorySanitizer unit tests.
add_custom_target(MsanUnitTests)
set_target_properties(MsanUnitTests PROPERTIES FOLDER "MSan unit tests")

# Adds MSan unit tests and benchmarks for architecture.
macro(add_msan_tests_for_arch arch kind)
  # Build gtest instrumented with MSan.
  set(MSAN_INST_GTEST)
  msan_compile(MSAN_INST_GTEST ${COMPILER_RT_GTEST_SOURCE} ${arch} "${kind}"
                               ${MSAN_UNITTEST_INSTRUMENTED_CFLAGS} ${ARGN})

  # Instrumented tests.
  set(MSAN_INST_TEST_OBJECTS)
  foreach (SOURCE ${MSAN_UNITTEST_SOURCES})
    msan_compile(MSAN_INST_TEST_OBJECTS ${SOURCE} ${arch} "${kind}"
                 ${MSAN_UNITTEST_INSTRUMENTED_CFLAGS} ${ARGN})
  endforeach(SOURCE)

  # Instrumented loadable module objects.
  set(MSAN_INST_LOADABLE_OBJECTS)
  msan_compile(MSAN_INST_LOADABLE_OBJECTS ${MSAN_LOADABLE_SOURCE} ${arch} "${kind}"
               ${MSAN_UNITTEST_INSTRUMENTED_CFLAGS} "-fPIC" ${ARGN})

  # Instrumented loadable library tests.
  set(MSAN_LOADABLE_SO)
  msan_link_shared(MSAN_LOADABLE_SO "libmsan_loadable" ${arch} "${kind}"
                   OBJECTS ${MSAN_INST_LOADABLE_OBJECTS}
                   DEPS ${MSAN_INST_LOADABLE_OBJECTS})

  set(MSAN_TEST_OBJECTS ${MSAN_INST_TEST_OBJECTS} ${MSAN_INST_GTEST})
  set(MSAN_TEST_DEPS ${MSAN_TEST_OBJECTS} libcxx_msan_${arch}
                     ${MSAN_LOADABLE_SO})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND MSAN_TEST_DEPS msan)
  endif()
  get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
  add_compiler_rt_test(MsanUnitTests "Msan-${arch}${kind}-Test" ${arch}
    OBJECTS ${MSAN_TEST_OBJECTS} ${MSAN_LIBCXX_SO}
    DEPS ${MSAN_TEST_DEPS}
    LINK_FLAGS ${MSAN_UNITTEST_LINK_FLAGS}
               ${TARGET_LINK_FLAGS}
               "-Wl,-rpath=${CMAKE_CURRENT_BINARY_DIR}"
               "-Wl,-rpath=${LIBCXX_PREFIX}/lib")
endmacro()

# We should only build MSan unit tests if we can build instrumented libcxx.
if(COMPILER_RT_CAN_EXECUTE_TESTS AND COMPILER_RT_HAS_LIBCXX_SOURCES)
  foreach(arch ${MSAN_SUPPORTED_ARCH})
    get_target_flags_for_arch(${arch} TARGET_CFLAGS)
    set(LIBCXX_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/../libcxx_msan_${arch})
    add_custom_libcxx(libcxx_msan_${arch} ${LIBCXX_PREFIX}
      DEPS ${MSAN_RUNTIME_LIBRARIES}
      CFLAGS ${MSAN_LIBCXX_CFLAGS} ${TARGET_CFLAGS})
    set(MSAN_LIBCXX_SO ${LIBCXX_PREFIX}/lib/libc++.so)

    add_msan_tests_for_arch(${arch} "")
    add_msan_tests_for_arch(${arch} "-with-call"
                            -mllvm -msan-instrumentation-with-call-threshold=0)
  endforeach()
endif()
