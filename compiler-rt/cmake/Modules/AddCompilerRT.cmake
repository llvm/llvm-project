include(AddLLVM)
include(ExternalProject)
include(CompilerRTUtils)

# Tries to add an "object library" target for a given list of OSs and/or
# architectures with name "<name>.<arch>" for non-Darwin platforms if
# architecture can be targeted, and "<name>.<os>" for Darwin platforms.
# add_compiler_rt_object_libraries(<name>
#                                  OS <os names>
#                                  ARCHS <architectures>
#                                  SOURCES <source files>
#                                  CFLAGS <compile flags>
#                                  DEFS <compile definitions>)
function(add_compiler_rt_object_libraries name)
  cmake_parse_arguments(LIB "" "" "OS;ARCHS;SOURCES;CFLAGS;DEFS" ${ARGN})
  set(libnames)
  if(APPLE)
    foreach(os ${LIB_OS})
      set(libname "${name}.${os}")
      set(libnames ${libnames} ${libname})
      set(extra_cflags_${libname} ${DARWIN_${os}_CFLAGS})
    endforeach()
  else()
    foreach(arch ${LIB_ARCHS})
      set(libname "${name}.${arch}")
      set(libnames ${libnames} ${libname})
      set(extra_cflags_${libname} ${TARGET_${arch}_CFLAGS})
      if(NOT CAN_TARGET_${arch})
        message(FATAL_ERROR "Archtecture ${arch} can't be targeted")
        return()
      endif()
    endforeach()
  endif()
  
  foreach(libname ${libnames})
    add_library(${libname} OBJECT ${LIB_SOURCES})
    set_target_compile_flags(${libname}
      ${CMAKE_CXX_FLAGS} ${extra_cflags_${libname}} ${LIB_CFLAGS})
    set_property(TARGET ${libname} APPEND PROPERTY
      COMPILE_DEFINITIONS ${LIB_DEFS})
    if(APPLE)
      set_target_properties(${libname} PROPERTIES OSX_ARCHITECTURES "${LIB_ARCHS}")
    endif()
  endforeach()
endfunction()

# Adds static or shared runtime for a given architecture and puts it in the
# proper directory in the build and install trees.
# add_compiler_rt_runtime(<name> <arch> {STATIC,SHARED}
#                         SOURCES <source files>
#                         CFLAGS <compile flags>
#                         DEFS <compile definitions>
#                         OUTPUT_NAME <output library name>)
macro(add_compiler_rt_runtime name arch type)
  if(CAN_TARGET_${arch})
    cmake_parse_arguments(LIB "" "OUTPUT_NAME" "SOURCES;CFLAGS;LINKFLAGS;DEFS" ${ARGN})
    add_library(${name} ${type} ${LIB_SOURCES})
    # Setup compile flags and definitions.
    set_target_compile_flags(${name}
      ${TARGET_${arch}_CFLAGS} ${LIB_CFLAGS})
    set_target_link_flags(${name}
      ${TARGET_${arch}_CFLAGS} ${LIB_CFLAGS} ${LIB_LINKFLAGS})
    set_property(TARGET ${name} APPEND PROPERTY
      COMPILE_DEFINITIONS ${LIB_DEFS})
    # Setup correct output directory in the build tree.
    set_target_properties(${name} PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR}
      LIBRARY_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR}
      RUNTIME_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
    if ("${LIB_OUTPUT_NAME}" STREQUAL "")
      set_target_properties(${name} PROPERTIES
        OUTPUT_NAME ${name}${COMPILER_RT_OS_SUFFIX})
    else()
      set_target_properties(${name} PROPERTIES
        OUTPUT_NAME ${LIB_OUTPUT_NAME})
    endif()
    # Add installation command.
    install(TARGETS ${name}
      ARCHIVE DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR}
      LIBRARY DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR}
      RUNTIME DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
  else()
    message(FATAL_ERROR "Archtecture ${arch} can't be targeted")
  endif()
endmacro()

# Same as add_compiler_rt_runtime(... STATIC), but creates a universal library
# for several architectures.
# add_compiler_rt_osx_static_runtime(<name> ARCHS <architectures>
#                                    SOURCES <source files>
#                                    CFLAGS <compile flags>
#                                    DEFS <compile definitions>)
macro(add_compiler_rt_osx_static_runtime name)
  cmake_parse_arguments(LIB "" "" "ARCHS;SOURCES;CFLAGS;DEFS" ${ARGN})
  add_library(${name} STATIC ${LIB_SOURCES})
  set_target_compile_flags(${name} ${LIB_CFLAGS})
  set_property(TARGET ${name} APPEND PROPERTY
    COMPILE_DEFINITIONS ${LIB_DEFS})
  set_target_properties(${name} PROPERTIES
    OSX_ARCHITECTURES "${LIB_ARCHS}"
    ARCHIVE_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
  install(TARGETS ${name}
    ARCHIVE DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
endmacro()

# Adds dynamic runtime library on osx/iossim, which supports multiple
# architectures.
# add_compiler_rt_darwin_dynamic_runtime(<name> <os>
#                                        ARCHS <architectures>
#                                        SOURCES <source files>
#                                        CFLAGS <compile flags>
#                                        DEFS <compile definitions>
#                                        LINKFLAGS <link flags>)
macro(add_compiler_rt_darwin_dynamic_runtime name os)
  cmake_parse_arguments(LIB "" "" "ARCHS;SOURCES;CFLAGS;DEFS;LINKFLAGS" ${ARGN})
  add_library(${name} SHARED ${LIB_SOURCES})
  set_target_compile_flags(${name} ${LIB_CFLAGS} ${DARWIN_${os}_CFLAGS})
  set_target_link_flags(${name} ${LIB_LINKFLAGS} ${DARWIN_${os}_LINKFLAGS})
  set_property(TARGET ${name} APPEND PROPERTY
    COMPILE_DEFINITIONS ${LIB_DEFS})
  set_target_properties(${name} PROPERTIES
    OSX_ARCHITECTURES "${LIB_ARCHS}"
    LIBRARY_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
  install(TARGETS ${name}
    LIBRARY DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
endmacro()

set(COMPILER_RT_TEST_CFLAGS)

# Unittests support.
set(COMPILER_RT_GTEST_PATH ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest)
set(COMPILER_RT_GTEST_SOURCE ${COMPILER_RT_GTEST_PATH}/src/gtest-all.cc)
set(COMPILER_RT_GTEST_CFLAGS
  -DGTEST_NO_LLVM_RAW_OSTREAM=1
  -DGTEST_HAS_RTTI=0
  -I${COMPILER_RT_GTEST_PATH}/include
  -I${COMPILER_RT_GTEST_PATH}
)

append_list_if(COMPILER_RT_DEBUG -DSANITIZER_DEBUG=1 COMPILER_RT_TEST_CFLAGS)

if(MSVC)
  # clang doesn't support exceptions on Windows yet.
  list(APPEND COMPILER_RT_TEST_CFLAGS -D_HAS_EXCEPTIONS=0)

  # We should teach clang to understand "#pragma intrinsic", see PR19898.
  list(APPEND COMPILER_RT_TEST_CFLAGS -Wno-undefined-inline)

  # Clang doesn't support SEH on Windows yet.
  list(APPEND COMPILER_RT_GTEST_CFLAGS -DGTEST_HAS_SEH=0)

  # gtest use a lot of stuff marked as deprecated on Windows.
  list(APPEND COMPILER_RT_GTEST_CFLAGS -Wno-deprecated-declarations)

  # Visual Studio 2012 only supports up to 8 template parameters in
  # std::tr1::tuple by default, but gtest requires 10
  if(MSVC_VERSION EQUAL 1700)
    list(APPEND COMPILER_RT_GTEST_CFLAGS -D_VARIADIC_MAX=10)
  endif()
endif()

# Link objects into a single executable with COMPILER_RT_TEST_COMPILER,
# using specified link flags. Make executable a part of provided
# test_suite.
# add_compiler_rt_test(<test_suite> <test_name>
#                      SUBDIR <subdirectory for binary>
#                      OBJECTS <object files>
#                      DEPS <deps (e.g. runtime libs)>
#                      LINK_FLAGS <link flags>)
macro(add_compiler_rt_test test_suite test_name)
  cmake_parse_arguments(TEST "" "SUBDIR" "OBJECTS;DEPS;LINK_FLAGS" "" ${ARGN})
  if(TEST_SUBDIR)
    set(output_bin "${CMAKE_CURRENT_BINARY_DIR}/${TEST_SUBDIR}/${test_name}")
  else()
    set(output_bin "${CMAKE_CURRENT_BINARY_DIR}/${test_name}")
  endif()
  if(MSVC)
    set(output_bin "${output_bin}.exe")
  endif()
  # Use host compiler in a standalone build, and just-built Clang otherwise.
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND TEST_DEPS clang)
  endif()
  # If we're not on MSVC, include the linker flags from CMAKE but override them
  # with the provided link flags. This ensures that flags which are required to
  # link programs at all are included, but the changes needed for the test
  # trump. With MSVC we can't do that because CMake is set up to run link.exe
  # when linking, not the compiler. Here, we hack it to use the compiler
  # because we want to use -fsanitize flags.
  if(NOT MSVC)
    set(TEST_LINK_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TEST_LINK_FLAGS}")
    separate_arguments(TEST_LINK_FLAGS)
  endif()
  add_custom_target(${test_name}
    COMMAND ${COMPILER_RT_TEST_COMPILER} ${TEST_OBJECTS}
            -o "${output_bin}"
            ${TEST_LINK_FLAGS}
    DEPENDS ${TEST_DEPS})
  # Make the test suite depend on the binary.
  add_dependencies(${test_suite} ${test_name})
endmacro()

macro(add_compiler_rt_resource_file target_name file_name)
  set(src_file "${CMAKE_CURRENT_SOURCE_DIR}/${file_name}")
  set(dst_file "${COMPILER_RT_OUTPUT_DIR}/${file_name}")
  add_custom_command(OUTPUT ${dst_file}
    DEPENDS ${src_file}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_file} ${dst_file}
    COMMENT "Copying ${file_name}...")
  add_custom_target(${target_name} DEPENDS ${dst_file})
  # Install in Clang resource directory.
  install(FILES ${file_name} DESTINATION ${COMPILER_RT_INSTALL_PATH})
endmacro()

macro(add_compiler_rt_script name)
  set(dst ${COMPILER_RT_EXEC_OUTPUT_DIR}/${name})
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/${name})
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying ${name}...")
  add_custom_target(${name} DEPENDS ${dst})
  install(FILES ${dst}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION ${COMPILER_RT_INSTALL_PATH}/bin)
endmacro(add_compiler_rt_script src name)

# Builds custom version of libc++ and installs it in <prefix>.
# Can be used to build sanitized versions of libc++ for running unit tests.
# add_custom_libcxx(<name> <prefix>
#                   DEPS <list of build deps>
#                   CFLAGS <list of compile flags>)
macro(add_custom_libcxx name prefix)
  if(NOT COMPILER_RT_HAS_LIBCXX_SOURCES)
    message(FATAL_ERROR "libcxx not found!")
  endif()

  cmake_parse_arguments(LIBCXX "" "" "DEPS;CFLAGS" ${ARGN})
  foreach(flag ${LIBCXX_CFLAGS})
    set(flagstr "${flagstr} ${flag}")
  endforeach()
  set(LIBCXX_CFLAGS ${flagstr})

  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND LIBCXX_DEPS clang)
  endif()

  ExternalProject_Add(${name}
    PREFIX ${prefix}
    SOURCE_DIR ${COMPILER_RT_LIBCXX_PATH}
    CMAKE_ARGS -DCMAKE_C_COMPILER=${COMPILER_RT_TEST_COMPILER}
               -DCMAKE_CXX_COMPILER=${COMPILER_RT_TEST_COMPILER}
               -DCMAKE_C_FLAGS=${LIBCXX_CFLAGS}
               -DCMAKE_CXX_FLAGS=${LIBCXX_CFLAGS}
               -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    LOG_BUILD 1
    LOG_CONFIGURE 1
    LOG_INSTALL 1
    )
  set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL TRUE)

  ExternalProject_Add_Step(${name} force-reconfigure
    DEPENDERS configure
    ALWAYS 1
    )

  ExternalProject_Add_Step(${name} clobber
    COMMAND ${CMAKE_COMMAND} -E remove_directory <BINARY_DIR>
    COMMAND ${CMAKE_COMMAND} -E make_directory <BINARY_DIR>
    COMMENT "Clobberring ${name} build directory..."
    DEPENDERS configure
    DEPENDS ${LIBCXX_DEPS}
    )
endmacro()
