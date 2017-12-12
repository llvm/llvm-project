include(ExternalProject)
include(CompilerRTUtils)

function(set_target_output_directories target output_dir)
  # For RUNTIME_OUTPUT_DIRECTORY variable, Multi-configuration generators
  # append a per-configuration subdirectory to the specified directory.
  # To avoid the appended folder, the configuration specific variable must be
  # set 'RUNTIME_OUTPUT_DIRECTORY_${CONF}':
  # RUNTIME_OUTPUT_DIRECTORY_DEBUG, RUNTIME_OUTPUT_DIRECTORY_RELEASE, ...
  if(CMAKE_CONFIGURATION_TYPES)
    foreach(build_mode ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER "${build_mode}" CONFIG_SUFFIX)
      set_target_properties("${target}" PROPERTIES
          "ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${output_dir}
          "LIBRARY_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${output_dir}
          "RUNTIME_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${output_dir})
    endforeach()
  else()
    set_target_properties("${target}" PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${output_dir}
        LIBRARY_OUTPUT_DIRECTORY ${output_dir}
        RUNTIME_OUTPUT_DIRECTORY ${output_dir})
  endif()
endfunction()

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
      list_intersect(LIB_ARCHS_${libname} DARWIN_${os}_ARCHS LIB_ARCHS)
    endforeach()
  else()
    foreach(arch ${LIB_ARCHS})
      set(libname "${name}.${arch}")
      set(libnames ${libnames} ${libname})
      set(extra_cflags_${libname} ${TARGET_${arch}_CFLAGS})
      if(NOT CAN_TARGET_${arch})
        message(FATAL_ERROR "Architecture ${arch} can't be targeted")
        return()
      endif()
    endforeach()
  endif()

  foreach(libname ${libnames})
    add_library(${libname} OBJECT ${LIB_SOURCES})

    # Strip out -msse3 if this isn't macOS.
    set(target_flags ${LIB_CFLAGS})
    if(APPLE AND NOT "${libname}" MATCHES ".*\.osx.*")
      list(REMOVE_ITEM target_flags "-msse3")
    endif()

    set_target_compile_flags(${libname}
      ${CMAKE_CXX_FLAGS} ${extra_cflags_${libname}} ${target_flags})
    set_property(TARGET ${libname} APPEND PROPERTY
      COMPILE_DEFINITIONS ${LIB_DEFS})
    set_target_properties(${libname} PROPERTIES FOLDER "Compiler-RT Libraries")
    if(APPLE)
      set_target_properties(${libname} PROPERTIES
        OSX_ARCHITECTURES "${LIB_ARCHS_${libname}}")
    endif()
  endforeach()
endfunction()

# Takes a list of object library targets, and a suffix and appends the proper
# TARGET_OBJECTS string to the output variable.
# format_object_libs(<output> <suffix> ...)
macro(format_object_libs output suffix)
  foreach(lib ${ARGN})
    list(APPEND ${output} $<TARGET_OBJECTS:${lib}.${suffix}>)
  endforeach()
endmacro()

function(add_compiler_rt_component name)
  add_custom_target(${name})
  set_target_properties(${name} PROPERTIES FOLDER "Compiler-RT Misc")
  if(COMMAND runtime_register_component)
    runtime_register_component(${name})
  endif()
  add_dependencies(compiler-rt ${name})
endfunction()

function(add_asm_sources output)
  set(${output} ${ARGN} PARENT_SCOPE)
  # Xcode will try to compile asm files as C ('clang -x c'), and that will fail.
  if (${CMAKE_GENERATOR} STREQUAL "Xcode")
    enable_language(ASM)
  else()
    # Pass ASM file directly to the C++ compiler.
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE C)
  endif()
endfunction()

macro(set_output_name output name arch)
  if(ANDROID AND ${arch} STREQUAL "i386")
    set(${output} "${name}-i686${COMPILER_RT_OS_SUFFIX}")
  else()
    set(${output} "${name}-${arch}${COMPILER_RT_OS_SUFFIX}")
  endif()
endmacro()

# Adds static or shared runtime for a list of architectures and operating
# systems and puts it in the proper directory in the build and install trees.
# add_compiler_rt_runtime(<name>
#                         {STATIC|SHARED}
#                         ARCHS <architectures>
#                         OS <os list>
#                         SOURCES <source files>
#                         CFLAGS <compile flags>
#                         LINK_FLAGS <linker flags>
#                         DEFS <compile definitions>
#                         LINK_LIBS <linked libraries> (only for shared library)
#                         OBJECT_LIBS <object libraries to use as sources>
#                         PARENT_TARGET <convenience parent target>)
function(add_compiler_rt_runtime name type)
  if(NOT type MATCHES "^(STATIC|SHARED)$")
    message(FATAL_ERROR "type argument must be STATIC or SHARED")
    return()
  endif()
  cmake_parse_arguments(LIB
    ""
    "PARENT_TARGET"
    "OS;ARCHS;SOURCES;CFLAGS;LINK_FLAGS;DEFS;LINK_LIBS;OBJECT_LIBS"
    ${ARGN})
  set(libnames)
  # Until we support this some other way, build compiler-rt runtime without LTO
  # to allow non-LTO projects to link with it.
  if(COMPILER_RT_HAS_FNO_LTO_FLAG)
    set(NO_LTO_FLAGS "-fno-lto")
  else()
    set(NO_LTO_FLAGS "")
  endif()

  if(APPLE)
    foreach(os ${LIB_OS})
      # Strip out -msse3 if this isn't macOS.
      list(LENGTH LIB_CFLAGS HAS_EXTRA_CFLAGS)
      if(HAS_EXTRA_CFLAGS AND NOT "${os}" MATCHES "^(osx)$")
        list(REMOVE_ITEM LIB_CFLAGS "-msse3")
      endif()
      if(type STREQUAL "STATIC")
        set(libname "${name}_${os}")
      else()
        set(libname "${name}_${os}_dynamic")
        set(extra_link_flags_${libname} ${DARWIN_${os}_LINK_FLAGS} ${LIB_LINK_FLAGS})
      endif()
      list_intersect(LIB_ARCHS_${libname} DARWIN_${os}_ARCHS LIB_ARCHS)
      if(LIB_ARCHS_${libname})
        list(APPEND libnames ${libname})
        set(extra_cflags_${libname} ${DARWIN_${os}_CFLAGS} ${NO_LTO_FLAGS} ${LIB_CFLAGS})
        set(output_name_${libname} ${libname}${COMPILER_RT_OS_SUFFIX})
        set(sources_${libname} ${LIB_SOURCES})
        format_object_libs(sources_${libname} ${os} ${LIB_OBJECT_LIBS})
      endif()
    endforeach()
  else()
    foreach(arch ${LIB_ARCHS})
      if(NOT CAN_TARGET_${arch})
        message(FATAL_ERROR "Architecture ${arch} can't be targeted")
        return()
      endif()
      if(type STREQUAL "STATIC")
        set(libname "${name}-${arch}")
        set_output_name(output_name_${libname} ${name} ${arch})
      else()
        set(libname "${name}-dynamic-${arch}")
        set(extra_cflags_${libname} ${TARGET_${arch}_CFLAGS} ${LIB_CFLAGS})
        set(extra_link_flags_${libname} ${TARGET_${arch}_LINK_FLAGS} ${LIB_LINK_FLAGS})
        if(WIN32)
          set_output_name(output_name_${libname} ${name}_dynamic ${arch})
        else()
          set_output_name(output_name_${libname} ${name} ${arch})
        endif()
      endif()
      set(sources_${libname} ${LIB_SOURCES})
      format_object_libs(sources_${libname} ${arch} ${LIB_OBJECT_LIBS})
      set(libnames ${libnames} ${libname})
      set(extra_cflags_${libname} ${TARGET_${arch}_CFLAGS} ${NO_LTO_FLAGS} ${LIB_CFLAGS})
    endforeach()
  endif()

  if(NOT libnames)
    return()
  endif()

  if(LIB_PARENT_TARGET)
    # If the parent targets aren't created we should create them
    if(NOT TARGET ${LIB_PARENT_TARGET})
      add_custom_target(${LIB_PARENT_TARGET})
    endif()
    if(NOT TARGET install-${LIB_PARENT_TARGET})
      # The parent install target specifies the parent component to scrape up
      # anything not installed by the individual install targets, and to handle
      # installation when running the multi-configuration generators.
      add_custom_target(install-${LIB_PARENT_TARGET}
                        DEPENDS ${LIB_PARENT_TARGET}
                        COMMAND "${CMAKE_COMMAND}"
                                -DCMAKE_INSTALL_COMPONENT=${LIB_PARENT_TARGET}
                                -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
      add_custom_target(install-${LIB_PARENT_TARGET}-stripped
                        DEPENDS ${LIB_PARENT_TARGET}
                        COMMAND "${CMAKE_COMMAND}"
                                -DCMAKE_INSTALL_COMPONENT=${LIB_PARENT_TARGET}
                                -DCMAKE_INSTALL_DO_STRIP=1
                                -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
      set_target_properties(install-${LIB_PARENT_TARGET} PROPERTIES
                            FOLDER "Compiler-RT Misc")
      set_target_properties(install-${LIB_PARENT_TARGET}-stripped PROPERTIES
                            FOLDER "Compiler-RT Misc")
      add_dependencies(install-compiler-rt install-${LIB_PARENT_TARGET})
      add_dependencies(install-compiler-rt-stripped install-${LIB_PARENT_TARGET}-stripped)
    endif()
  endif()

  foreach(libname ${libnames})
    # If you are using a multi-configuration generator we don't generate
    # per-library install rules, so we fall back to the parent target COMPONENT
    if(CMAKE_CONFIGURATION_TYPES AND LIB_PARENT_TARGET)
      set(COMPONENT_OPTION COMPONENT ${LIB_PARENT_TARGET})
    else()
      set(COMPONENT_OPTION COMPONENT ${libname})
    endif()

    add_library(${libname} ${type} ${sources_${libname}})
    set_target_compile_flags(${libname} ${extra_cflags_${libname}})
    set_target_link_flags(${libname} ${extra_link_flags_${libname}})
    set_property(TARGET ${libname} APPEND PROPERTY
                COMPILE_DEFINITIONS ${LIB_DEFS})
    set_target_output_directories(${libname} ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
    set_target_properties(${libname} PROPERTIES
        OUTPUT_NAME ${output_name_${libname}})
    set_target_properties(${libname} PROPERTIES FOLDER "Compiler-RT Runtime")
    if(LIB_LINK_LIBS)
      target_link_libraries(${libname} ${LIB_LINK_LIBS})
    endif()
    if(${type} STREQUAL "SHARED")
      if(WIN32 AND NOT CYGWIN AND NOT MINGW)
        set_target_properties(${libname} PROPERTIES IMPORT_PREFIX "")
        set_target_properties(${libname} PROPERTIES IMPORT_SUFFIX ".lib")
      endif()
      if(APPLE)
        # Ad-hoc sign the dylibs
        add_custom_command(TARGET ${libname}
          POST_BUILD  
          COMMAND codesign --sign - $<TARGET_FILE:${libname}>
          WORKING_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR}
        )
      endif()
    endif()
    install(TARGETS ${libname}
      ARCHIVE DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR}
              ${COMPONENT_OPTION}
      LIBRARY DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR}
              ${COMPONENT_OPTION}
      RUNTIME DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR}
              ${COMPONENT_OPTION})

    # We only want to generate per-library install targets if you aren't using
    # an IDE because the extra targets get cluttered in IDEs.
    if(NOT CMAKE_CONFIGURATION_TYPES)
      add_custom_target(install-${libname}
                        DEPENDS ${libname}
                        COMMAND "${CMAKE_COMMAND}"
                                -DCMAKE_INSTALL_COMPONENT=${libname}
                                -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
      add_custom_target(install-${libname}-stripped
                        DEPENDS ${libname}
                        COMMAND "${CMAKE_COMMAND}"
                                -DCMAKE_INSTALL_COMPONENT=${libname}
                                -DCMAKE_INSTALL_DO_STRIP=1
                                -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
      # If you have a parent target specified, we bind the new install target
      # to the parent install target.
      if(LIB_PARENT_TARGET)
        add_dependencies(install-${LIB_PARENT_TARGET} install-${libname})
        add_dependencies(install-${LIB_PARENT_TARGET}-stripped install-${libname}-stripped)
      endif()
    endif()
    if(APPLE)
      set_target_properties(${libname} PROPERTIES
      OSX_ARCHITECTURES "${LIB_ARCHS_${libname}}")
    endif()

    if(type STREQUAL "SHARED")
      rt_externalize_debuginfo(${libname})
    endif()
  endforeach()
  if(LIB_PARENT_TARGET)
    add_dependencies(${LIB_PARENT_TARGET} ${libnames})
  endif()
endfunction()

# when cross compiling, COMPILER_RT_TEST_COMPILER_CFLAGS help
# in compilation and linking of unittests.
string(REPLACE " " ";" COMPILER_RT_UNITTEST_CFLAGS "${COMPILER_RT_TEST_COMPILER_CFLAGS}")
set(COMPILER_RT_UNITTEST_LINK_FLAGS ${COMPILER_RT_UNITTEST_CFLAGS})

# Unittests support.
set(COMPILER_RT_GTEST_PATH ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest)
set(COMPILER_RT_GTEST_SOURCE ${COMPILER_RT_GTEST_PATH}/src/gtest-all.cc)
set(COMPILER_RT_GTEST_CFLAGS
  -DGTEST_NO_LLVM_RAW_OSTREAM=1
  -DGTEST_HAS_RTTI=0
  -I${COMPILER_RT_GTEST_PATH}/include
  -I${COMPILER_RT_GTEST_PATH}
)

append_list_if(COMPILER_RT_DEBUG -DSANITIZER_DEBUG=1 COMPILER_RT_UNITTEST_CFLAGS)
append_list_if(COMPILER_RT_HAS_WCOVERED_SWITCH_DEFAULT_FLAG -Wno-covered-switch-default COMPILER_RT_UNITTEST_CFLAGS)

if(MSVC)
  # clang doesn't support exceptions on Windows yet.
  list(APPEND COMPILER_RT_UNITTEST_CFLAGS -D_HAS_EXCEPTIONS=0)

  # We should teach clang to understand "#pragma intrinsic", see PR19898.
  list(APPEND COMPILER_RT_UNITTEST_CFLAGS -Wno-undefined-inline)

  # Clang doesn't support SEH on Windows yet.
  list(APPEND COMPILER_RT_GTEST_CFLAGS -DGTEST_HAS_SEH=0)

  # gtest use a lot of stuff marked as deprecated on Windows.
  list(APPEND COMPILER_RT_GTEST_CFLAGS -Wno-deprecated-declarations)
endif()

# Compile and register compiler-rt tests.
# generate_compiler_rt_tests(<output object files> <test_suite> <test_name>
#                           <test architecture>
#                           KIND <custom prefix>
#                           SUBDIR <subdirectory for testing binary>
#                           SOURCES <sources to compile>
#                           RUNTIME <tests runtime to link in>
#                           CFLAGS <compile-time flags>
#                           COMPILE_DEPS <compile-time dependencies>
#                           DEPS <dependencies>
#                           LINK_FLAGS <flags to use during linking>
# )
function(generate_compiler_rt_tests test_objects test_suite testname arch)
  cmake_parse_arguments(TEST "" "KIND;RUNTIME;SUBDIR"
    "SOURCES;COMPILE_DEPS;DEPS;CFLAGS;LINK_FLAGS" ${ARGN})

  foreach(source ${TEST_SOURCES})
    sanitizer_test_compile(
      "${test_objects}" "${source}" "${arch}"
      KIND ${TEST_KIND}
      COMPILE_DEPS ${TEST_COMPILE_DEPS}
      DEPS ${TEST_DEPS}
      CFLAGS ${TEST_CFLAGS}
      )
  endforeach()

  set(TEST_DEPS ${${test_objects}})

  if(NOT "${TEST_RUNTIME}" STREQUAL "")
    list(APPEND TEST_DEPS ${TEST_RUNTIME})
    list(APPEND "${test_objects}" $<TARGET_FILE:${TEST_RUNTIME}>)
  endif()

  add_compiler_rt_test(${test_suite} "${testname}" "${arch}"
    SUBDIR ${TEST_SUBDIR}
    OBJECTS ${${test_objects}}
    DEPS ${TEST_DEPS}
    LINK_FLAGS ${TEST_LINK_FLAGS}
    )
  set("${test_objects}" "${${test_objects}}" PARENT_SCOPE)
endfunction()

# Link objects into a single executable with COMPILER_RT_TEST_COMPILER,
# using specified link flags. Make executable a part of provided
# test_suite.
# add_compiler_rt_test(<test_suite> <test_name> <arch>
#                      SUBDIR <subdirectory for binary>
#                      OBJECTS <object files>
#                      DEPS <deps (e.g. runtime libs)>
#                      LINK_FLAGS <link flags>)
function(add_compiler_rt_test test_suite test_name arch)
  cmake_parse_arguments(TEST "" "SUBDIR" "OBJECTS;DEPS;LINK_FLAGS" "" ${ARGN})
  set(output_dir ${CMAKE_CURRENT_BINARY_DIR})
  if(TEST_SUBDIR)
    set(output_dir "${output_dir}/${TEST_SUBDIR}")
  endif()
  set(output_dir "${output_dir}/${CMAKE_CFG_INTDIR}")
  file(MAKE_DIRECTORY "${output_dir}")
  set(output_bin "${output_dir}/${test_name}")
  if(MSVC)
    set(output_bin "${output_bin}.exe")
  endif()

  # Use host compiler in a standalone build, and just-built Clang otherwise.
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND TEST_DEPS clang)
  endif()

  get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
  list(APPEND TEST_LINK_FLAGS ${TARGET_LINK_FLAGS})

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
  add_custom_command(
    OUTPUT "${output_bin}"
    COMMAND ${COMPILER_RT_TEST_COMPILER} ${TEST_OBJECTS} -o "${output_bin}"
            ${TEST_LINK_FLAGS}
    DEPENDS ${TEST_DEPS}
    )
  add_custom_target(T${test_name} DEPENDS "${output_bin}")
  set_target_properties(T${test_name} PROPERTIES FOLDER "Compiler-RT Tests")

  # Make the test suite depend on the binary.
  add_dependencies(${test_suite} T${test_name})
endfunction()

macro(add_compiler_rt_resource_file target_name file_name component)
  set(src_file "${CMAKE_CURRENT_SOURCE_DIR}/${file_name}")
  set(dst_file "${COMPILER_RT_OUTPUT_DIR}/${file_name}")
  add_custom_command(OUTPUT ${dst_file}
    DEPENDS ${src_file}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_file} ${dst_file}
    COMMENT "Copying ${file_name}...")
  add_custom_target(${target_name} DEPENDS ${dst_file})
  # Install in Clang resource directory.
  install(FILES ${file_name}
    DESTINATION ${COMPILER_RT_INSTALL_PATH}
    COMPONENT ${component})
  add_dependencies(${component} ${target_name})

  set_target_properties(${target_name} PROPERTIES FOLDER "Compiler-RT Misc")
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
  if(NOT COMPILER_RT_LIBCXX_PATH)
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
    CMAKE_ARGS -DCMAKE_MAKE_PROGRAM:STRING=${CMAKE_MAKE_PROGRAM}
               -DCMAKE_C_COMPILER=${COMPILER_RT_TEST_COMPILER}
               -DCMAKE_CXX_COMPILER=${COMPILER_RT_TEST_CXX_COMPILER}
               -DCMAKE_C_FLAGS=${LIBCXX_CFLAGS}
               -DCMAKE_CXX_FLAGS=${LIBCXX_CFLAGS}
               -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
               -DLLVM_PATH=${LLVM_MAIN_SRC_DIR}
               -DLIBCXX_STANDALONE_BUILD=On
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

function(rt_externalize_debuginfo name)
  if(NOT COMPILER_RT_EXTERNALIZE_DEBUGINFO)
    return()
  endif()

  if(NOT COMPILER_RT_EXTERNALIZE_DEBUGINFO_SKIP_STRIP)
    set(strip_command COMMAND xcrun strip -Sl $<TARGET_FILE:${name}>)
  endif()

  if(APPLE)
    if(CMAKE_CXX_FLAGS MATCHES "-flto"
      OR CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE} MATCHES "-flto")

      set(lto_object ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}-lto.o)
      set_property(TARGET ${name} APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,-object_path_lto -Wl,${lto_object}")
    endif()
    add_custom_command(TARGET ${name} POST_BUILD
      COMMAND xcrun dsymutil $<TARGET_FILE:${name}>
      ${strip_command})
  else()
    message(FATAL_ERROR "COMPILER_RT_EXTERNALIZE_DEBUGINFO isn't implemented for non-darwin platforms!")
  endif()
endfunction()


# Configure lit configuration files, including compiler-rt specific variables.
function(configure_compiler_rt_lit_site_cfg input output)
  set_llvm_build_mode()

  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} COMPILER_RT_RESOLVED_TEST_COMPILER ${COMPILER_RT_TEST_COMPILER})
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} COMPILER_RT_RESOLVED_LIBRARY_OUTPUT_DIR ${COMPILER_RT_LIBRARY_OUTPUT_DIR})

  configure_lit_site_cfg(${input} ${output})
endfunction()
