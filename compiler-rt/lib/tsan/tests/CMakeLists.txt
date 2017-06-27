include_directories(../rtl)

add_custom_target(TsanUnitTests)
set_target_properties(TsanUnitTests PROPERTIES
  FOLDER "TSan unittests")

set(TSAN_UNITTEST_CFLAGS
  ${TSAN_CFLAGS}
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/tsan/rtl
  -DGTEST_HAS_RTTI=0)

if(APPLE)
  list(APPEND TSAN_UNITTEST_CFLAGS ${DARWIN_osx_CFLAGS})
endif()

set(TSAN_RTL_HEADERS)
foreach (header ${TSAN_HEADERS})
  list(APPEND TSAN_RTL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../${header})
endforeach()

# tsan_compile(obj_list, source, arch, {headers})
macro(tsan_compile obj_list source arch)
  get_filename_component(basename ${source} NAME)
  set(output_obj "${basename}.${arch}.o")
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  set(COMPILE_DEPS ${TSAN_RTL_HEADERS} ${ARGN})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND COMPILE_DEPS gtest tsan)
  endif()
  clang_compile(${output_obj} ${source}
          CFLAGS ${TSAN_UNITTEST_CFLAGS} ${TARGET_CFLAGS}
          DEPS ${COMPILE_DEPS})
  list(APPEND ${obj_list} ${output_obj})
endmacro()

macro(add_tsan_unittest testname)
  set(TSAN_TEST_ARCH ${TSAN_SUPPORTED_ARCH})
  if(APPLE)
    darwin_filter_host_archs(TSAN_SUPPORTED_ARCH TSAN_TEST_ARCH)
  endif()
  if(UNIX)
    foreach(arch ${TSAN_TEST_ARCH})
      cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
      set(TEST_OBJECTS)
      foreach(SOURCE ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE})
        tsan_compile(TEST_OBJECTS ${SOURCE} ${arch} ${TEST_HEADERS})
      endforeach()
      get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
      set(TEST_DEPS ${TEST_OBJECTS})
      if(NOT COMPILER_RT_STANDALONE_BUILD)
        list(APPEND TEST_DEPS tsan)
      endif()
      if(NOT APPLE)
        # FIXME: Looks like we should link TSan with just-built runtime,
        # and not rely on -fsanitize=thread, as these tests are essentially
        # unit tests.
        add_compiler_rt_test(TsanUnitTests ${testname}
                OBJECTS ${TEST_OBJECTS}
                DEPS ${TEST_DEPS}
                LINK_FLAGS ${TARGET_LINK_FLAGS}
                           -fsanitize=thread
                           -lstdc++ -lm)
      else()
        set(TSAN_TEST_RUNTIME_OBJECTS
          $<TARGET_OBJECTS:RTTsan_dynamic.osx>
          $<TARGET_OBJECTS:RTInterception.osx>
          $<TARGET_OBJECTS:RTSanitizerCommon.osx>
          $<TARGET_OBJECTS:RTSanitizerCommonLibc.osx>
          $<TARGET_OBJECTS:RTUbsan.osx>)
        set(TSAN_TEST_RUNTIME RTTsanTest.${testname}.${arch})
        add_library(${TSAN_TEST_RUNTIME} STATIC ${TSAN_TEST_RUNTIME_OBJECTS})
        set_target_properties(${TSAN_TEST_RUNTIME} PROPERTIES
          ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        list(APPEND TEST_OBJECTS lib${TSAN_TEST_RUNTIME}.a)
        list(APPEND TEST_DEPS ${TSAN_TEST_RUNTIME})

        add_weak_symbols("ubsan" WEAK_SYMBOL_LINK_FLAGS)
        add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

        # Intentionally do *not* link with `-fsanitize=thread`. We already link
        # against a static version of the runtime, and we don't want the dynamic
        # one.
        add_compiler_rt_test(TsanUnitTests "${testname}-${arch}-Test"
                OBJECTS ${TEST_OBJECTS}
                DEPS ${TEST_DEPS}
                LINK_FLAGS ${TARGET_LINK_FLAGS} ${DARWIN_osx_LINK_FLAGS}
                           ${WEAK_SYMBOL_LINK_FLAGS} -lc++)
      endif()
    endforeach()
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  add_subdirectory(rtl)
  add_subdirectory(unit)
endif()
