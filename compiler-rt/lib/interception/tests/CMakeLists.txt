include(CompilerRTCompile)

filter_available_targets(INTERCEPTION_UNITTEST_SUPPORTED_ARCH x86_64 i386 mips64 mips64el)

set(INTERCEPTION_UNITTESTS
  interception_linux_test.cc
  interception_test_main.cc
  interception_win_test.cc
)

set(INTERCEPTION_TEST_HEADERS)

set(INTERCEPTION_TEST_CFLAGS_COMMON
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/interception
  -fno-rtti
  -O2
  -Werror=sign-compare
  -Wno-non-virtual-dtor)

# -gline-tables-only must be enough for these tests, so use it if possible.
if(COMPILER_RT_TEST_COMPILER_ID MATCHES "Clang")
  list(APPEND INTERCEPTION_TEST_CFLAGS_COMMON -gline-tables-only)
else()
  list(APPEND INTERCEPTION_TEST_CFLAGS_COMMON -g)
endif()
if(MSVC)
  list(APPEND INTERCEPTION_TEST_CFLAGS_COMMON -gcodeview)
endif()
list(APPEND INTERCEPTION_TEST_LINK_FLAGS_COMMON -g)

if(NOT MSVC)
  list(APPEND INTERCEPTION_TEST_LINK_FLAGS_COMMON --driver-mode=g++)
endif()

if(ANDROID)
  list(APPEND INTERCEPTION_TEST_LINK_FLAGS_COMMON -pie)
endif()

set(INTERCEPTION_TEST_LINK_LIBS)
append_list_if(COMPILER_RT_HAS_LIBLOG log INTERCEPTION_TEST_LINK_LIBS)
# NDK r10 requires -latomic almost always.
append_list_if(ANDROID atomic INTERCEPTION_TEST_LINK_LIBS)

append_list_if(COMPILER_RT_HAS_LIBDL -ldl INTERCEPTION_TEST_LINK_FLAGS_COMMON)
append_list_if(COMPILER_RT_HAS_LIBRT -lrt INTERCEPTION_TEST_LINK_FLAGS_COMMON)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD -pthread INTERCEPTION_TEST_LINK_FLAGS_COMMON)
# x86_64 FreeBSD 9.2 additionally requires libc++ to build the tests. Also,
# 'libm' shall be specified explicitly to build i386 tests.
if(CMAKE_SYSTEM MATCHES "FreeBSD-9.2-RELEASE")
  list(APPEND INTERCEPTION_TEST_LINK_FLAGS_COMMON "-lc++ -lm")
endif()

include_directories(..)
include_directories(../..)

# Adds static library which contains interception object file
# (universal binary on Mac and arch-specific object files on Linux).
macro(add_interceptor_lib library)
  add_library(${library} STATIC ${ARGN})
  set_target_properties(${library} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT Runtime tests")
endmacro()

function(get_interception_lib_for_arch arch lib lib_name)
  if(APPLE)
    set(tgt_name "RTInterception.test.osx")
  else()
    set(tgt_name "RTInterception.test.${arch}")
  endif()
  set(${lib} "${tgt_name}" PARENT_SCOPE)
  if(CMAKE_CONFIGURATION_TYPES)
   set(configuration_path "${CMAKE_CFG_INTDIR}/")
  else()
   set(configuration_path "")
  endif()
  if(NOT MSVC)
    set(${lib_name} "${configuration_path}lib${tgt_name}.a" PARENT_SCOPE)
  else()
    set(${lib_name} "${configuration_path}${tgt_name}.lib" PARENT_SCOPE)
  endif()
endfunction()

# Interception unit tests testsuite.
add_custom_target(InterceptionUnitTests)
set_target_properties(InterceptionUnitTests PROPERTIES
  FOLDER "Compiler-RT Tests")

# Adds interception tests for architecture.
macro(add_interception_tests_for_arch arch)
  get_target_flags_for_arch(${arch} TARGET_FLAGS)
  set(INTERCEPTION_TEST_SOURCES ${INTERCEPTION_UNITTESTS}
                             ${COMPILER_RT_GTEST_SOURCE})
  set(INTERCEPTION_TEST_COMPILE_DEPS ${INTERCEPTION_TEST_HEADERS})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND INTERCEPTION_TEST_COMPILE_DEPS gtest)
  endif()
  set(INTERCEPTION_TEST_OBJECTS)
  foreach(source ${INTERCEPTION_TEST_SOURCES})
    get_filename_component(basename ${source} NAME)
    if(CMAKE_CONFIGURATION_TYPES)
      set(output_obj "${CMAKE_CFG_INTDIR}/${basename}.${arch}.o")
    else()
      set(output_obj "${basename}.${arch}.o")
    endif()
    clang_compile(${output_obj} ${source}
                  CFLAGS ${INTERCEPTION_TEST_CFLAGS_COMMON} ${TARGET_FLAGS}
                  DEPS ${INTERCEPTION_TEST_COMPILE_DEPS})
    list(APPEND INTERCEPTION_TEST_OBJECTS ${output_obj})
  endforeach()
  get_interception_lib_for_arch(${arch} INTERCEPTION_COMMON_LIB
                                INTERCEPTION_COMMON_LIB_NAME)
  # Add unittest target.
  set(INTERCEPTION_TEST_NAME "Interception-${arch}-Test")
  add_compiler_rt_test(InterceptionUnitTests ${INTERCEPTION_TEST_NAME}
                       OBJECTS ${INTERCEPTION_TEST_OBJECTS}
                               ${INTERCEPTION_COMMON_LIB_NAME}
                       DEPS ${INTERCEPTION_TEST_OBJECTS} ${INTERCEPTION_COMMON_LIB}
                       LINK_FLAGS ${INTERCEPTION_TEST_LINK_FLAGS_COMMON}
                                  ${TARGET_FLAGS})
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS AND NOT ANDROID AND NOT APPLE)
  # We use just-built clang to build interception unittests, so we must
  # be sure that produced binaries would work.
  if(APPLE)
    add_interceptor_lib("RTInterception.test.osx"
                        $<TARGET_OBJECTS:RTInterception.osx>)
  else()
    foreach(arch ${INTERCEPTION_UNITTEST_SUPPORTED_ARCH})
      add_interceptor_lib("RTInterception.test.${arch}"
                          $<TARGET_OBJECTS:RTInterception.${arch}>)
    endforeach()
  endif()
  foreach(arch ${INTERCEPTION_UNITTEST_SUPPORTED_ARCH})
    add_interception_tests_for_arch(${arch})
  endforeach()
endif()
