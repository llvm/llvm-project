function(add_sycl_rt_library LIB_NAME LIB_OBJ_NAME)
  # Add an optional argument so we can get the library name to
  # link with for Windows Debug version
  cmake_parse_arguments(ARG "" "IMPLIB_NAME" "COMPILE_OPTIONS;SOURCES" ${ARGN})

  add_library(${LIB_OBJ_NAME} OBJECT ${ARG_SOURCES})

  # Common compilation step setup
  target_compile_definitions(${LIB_OBJ_NAME} PRIVATE $<$<BOOL:${MSVC}>:__SYCL_BUILD_SYCL_DLL>)
  target_include_directories(
    ${LIB_OBJ_NAME}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${libsycl_inc_dir}"
  )

  add_library(${LIB_NAME} SHARED
              $<TARGET_OBJECTS:${LIB_OBJ_NAME}>)

  # Unlike for libsycl, for LLVMSupport we have only one version for a given build,
  # so, we link LLVMSupport lib to matching libsycl version only.
  if (LIBSYCL_ENABLE_STACK_PRINTING)
    if(NOT MSVC OR (CMAKE_BUILD_TYPE STREQUAL "Debug" AND ARG_COMPILE_OPTIONS MATCHES ".*MDd.*") OR
      (NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT ARG_COMPILE_OPTIONS MATCHES ".*MDd.*"))
        add_dependencies(${LIB_NAME} LLVMSupport)
        target_compile_definitions(${LIB_OBJ_NAME} PUBLIC ENABLE_STACK_TRACE)
        target_link_libraries(${LIB_NAME} PRIVATE LLVMSupport)
    endif()
  endif()

  # TODO: Enabled for MSVC
  if (NOT MSVC AND LIBSYCL_DEBUG_SYMBOLS)
    separate_arguments(CMAKE_CXX_FLAGS_DEBUG_SEPARATED UNIX_COMMAND "${CMAKE_CXX_FLAGS_DEBUG}")
    target_compile_options(${LIB_NAME} PRIVATE ${CMAKE_CXX_FLAGS_DEBUG_SEPARATED})
  endif()

  if (ARG_COMPILE_OPTIONS)
    target_compile_options(${LIB_OBJ_NAME} PRIVATE ${ARG_COMPILE_OPTIONS})
  endif()

  add_dependencies(${LIB_OBJ_NAME}
    sycl-headers
  )

  set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

  if (WIN32)
    target_link_libraries(${LIB_NAME} PRIVATE shlwapi)
    if (ARG_IMPLIB_NAME)
      add_custom_command(
        TARGET ${LIB_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${LIB_NAME}.lib ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${ARG_IMPLIB_NAME}.lib
        COMMENT "Creating version-agnostic copy of the import library.")
      install(
        FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${ARG_IMPLIB_NAME}.lib
        DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT libsycl)
    endif()
  endif()

  if (MSVC)
    # Install stripped PDB
    check_linker_flag(CXX "LINKER:/PDBSTRIPPED:${LIB_NAME}.stripped.pdb"
                          LINKER_SUPPORTS_PDBSTRIPPED)
    if(LINKER_SUPPORTS_PDBSTRIPPED)
      target_link_options(${LIB_NAME}
                          PRIVATE "LINKER:/PDBSTRIPPED:${LIB_NAME}.stripped.pdb")
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}.stripped.pdb"
              DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
              RENAME "${LIB_NAME}.pdb"
              COMPONENT ${LIB_NAME}
              OPTIONAL)
    endif()
  else()
    target_compile_options(${LIB_OBJ_NAME} PUBLIC
                           -fvisibility=hidden -fvisibility-inlines-hidden)

    # Sycl math built-in macros cause a GCC 4.6 'note' to be output repeatedly.
    # => note: the ABI for passing parameters with 32-byte alignment has changed in GCC 4.6
    # Seems to be no way to suppress it except use -Wno-psabi
    target_compile_options(${LIB_OBJ_NAME} PUBLIC -Wno-psabi)

    if (UNIX AND NOT APPLE)
      set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
      target_link_libraries(
          ${LIB_NAME} PRIVATE "-Wl,--version-script=${linker_script}")
      set_target_properties(${LIB_NAME} PROPERTIES LINK_DEPENDS ${linker_script})
    endif()
  endif()

  find_package(Threads REQUIRED)

  target_link_libraries(${LIB_NAME}
    PRIVATE
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
  )

  # Workaround for a bug in GCC version 5 and higher.
  # More information https://bugs.launchpad.net/ubuntu/+source/gcc-5/+bug/1568899
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    target_link_libraries(${ARGV0} PRIVATE gcc_s gcc)
  endif()

  set_target_properties(${LIB_NAME} PROPERTIES
                        VERSION ${LIBSYCL_VERSION_STRING}
                        SOVERSION ${LIBSYCL_VERSION_STRING})

  check_cxx_compiler_flag(-Winstantiation-after-specialization
    HAS_INST_AFTER_SPEC)
  if (HAS_INST_AFTER_SPEC)
    target_compile_options(${LIB_OBJ_NAME} PRIVATE
      -Winstantiation-after-specialization)
  endif()
endfunction(add_sycl_rt_library)

set(LIBSYCL_SOURCES
    "platform.cpp"
)

if (MSVC)
  # MSVC provides two incompatible build variants for its CRT: release and debug
  # To avoid potential issues in user code we also need to provide two kinds
  # of SYCL Runtime Library for release and debug configurations.
  foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "/MDd" "" ${flag_var} "${${flag_var}}")
    string(REGEX REPLACE "/MTd" "" ${flag_var} "${${flag_var}}")
    string(REGEX REPLACE "/MD" "" ${flag_var} "${${flag_var}}")
    string(REGEX REPLACE "/MT" "" ${flag_var} "${${flag_var}}")
  endforeach()

  set(WIN_DUPE "1")
  add_sycl_rt_library(sycl${LIBSYCL_MAJOR_VERSION}d sycld_object COMPILE_OPTIONS "/MDd" SOURCES ${LIBSYCL_SOURCES} IMPLIB_NAME sycld)
  unset(WIN_DUPE)
  add_library(sycld ALIAS sycl${LIBSYCL_MAJOR_VERSION}d)

  set(LIBSYCL_EXTRA_OPTS "/MD")
endif()

if (WIN32)
set(LIB_NAME "sycl${LIBSYCL_MAJOR_VERSION}")
else()
set(LIB_NAME "sycl")
endif()
# Version-agnostic name of the import library, has effect on Windows only.
set(IMPLIB_NAME "sycl")

add_sycl_rt_library(${LIB_NAME} sycl_object COMPILE_OPTIONS ${LIBSYCL_EXTRA_OPTS} SOURCES ${LIBSYCL_SOURCES} IMPLIB_NAME ${IMPLIB_NAME})

if (WIN32)
  add_library(sycl ALIAS ${LIB_NAME})
endif()

install(TARGETS ${LIBSYCL_RT_LIBS}
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT libsycl
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT libsycl
  RUNTIME DESTINATION "bin" COMPONENT libsycl)
