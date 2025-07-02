list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../runtimes/cmake/Modules")
include(WarningFlags)

function(add_sycl_rt_library LIB_TARGET_NAME LIB_OBJ_NAME LIB_OUTPUT_NAME)
  if (NOT LLVM_ENABLE_PIC)
    message( FATAL_ERROR "Position-Independent Code generation is required for libsycl shared library" )
  endif()
  # Add an optional argument so we can get the library name to
  # link with for Windows Debug version
  cmake_parse_arguments(ARG "" "IMPLIB_OUTPUT_NAME" "COMPILE_OPTIONS;SOURCES" ${ARGN})

  add_library(${LIB_OBJ_NAME} OBJECT ${ARG_SOURCES})

  # Common compilation step setup
  target_compile_definitions(${LIB_OBJ_NAME} PRIVATE
              $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:_LIBSYCL_BUILDING_LIBRARY>)
  cxx_add_warning_flags(${LIB_OBJ_NAME} ${LIBSYCL_ENABLE_WERROR} ${LIBSYCL_ENABLE_PEDANTIC})

  target_include_directories(
    ${LIB_OBJ_NAME}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${LIBSYCL_BUILD_INCLUDE_DIR}
  )

  add_library(${LIB_TARGET_NAME} SHARED
              $<TARGET_OBJECTS:${LIB_OBJ_NAME}>)

  add_dependencies(${LIB_OBJ_NAME}
    sycl-headers
  )

  set_target_properties(${LIB_TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

  if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_custom_command(
      TARGET ${LIB_TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${LIB_OUTPUT_NAME}.lib
              ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${ARG_IMPLIB_OUTPUT_NAME}.lib
      COMMENT "Creating version-agnostic copy of the import library.")
    install(
      FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${ARG_IMPLIB_OUTPUT_NAME}.lib
      DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT libsycl)
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    # Install stripped PDB
    set(PDB_FILENAME "${LIB_TARGET_NAME}$<$<CONFIG:Debug>:d>")
    check_linker_flag(CXX "LINKER:/PDBSTRIPPED:${PDB_FILENAME}.stripped.pdb"
                          LINKER_SUPPORTS_PDBSTRIPPED)
    if(LINKER_SUPPORTS_PDBSTRIPPED)
      target_link_options(${LIB_TARGET_NAME}
                          PRIVATE "LINKER:/PDBSTRIPPED:${PDB_FILENAME}.stripped.pdb")
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PDB_FILENAME}.stripped.pdb"
              DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
              RENAME "${PDB_FILENAME}.pdb"
              COMPONENT ${LIB_TARGET_NAME}
              OPTIONAL)
    endif()

    target_compile_options(${LIB_OBJ_NAME} PUBLIC /EHsc)
  else()
    target_compile_options(${LIB_OBJ_NAME} PUBLIC
                           -fvisibility=hidden -fvisibility-inlines-hidden)

    if (UNIX AND NOT APPLE)
      set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
      target_link_libraries(
          ${LIB_TARGET_NAME} PRIVATE "-Wl,--version-script=${linker_script}")
      set_target_properties(${LIB_TARGET_NAME} PROPERTIES LINK_DEPENDS ${linker_script})
    endif()
  endif()

  find_package(Threads REQUIRED)

  target_link_libraries(${LIB_TARGET_NAME}
    PRIVATE
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
  )

  set_target_properties(${LIB_TARGET_NAME} PROPERTIES
                        VERSION ${LIBSYCL_VERSION_STRING}
                        SOVERSION ${LIBSYCL_VERSION_STRING})
endfunction(add_sycl_rt_library)

set(LIBSYCL_SOURCES
    "platform.cpp"
)

# Version-agnostic name of the import library, has effect on Windows only.
set(IMPLIB_NAME "libsycl")

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
set(LIB_NAME "sycl${LIBSYCL_MAJOR_VERSION}")
set(LIB_OUTPUT_NAME "${LIB_NAME}")
else()
set(LIB_NAME "sycl")
set(LIB_OUTPUT_NAME "${LIB_NAME}")
  if (CMAKE_BUILD_TYPE STREQUAL Debug OR CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebugDLL)
    set(LIB_OUTPUT_NAME "${LIB_OUTPUT_NAME}d")
    set(IMPLIB_NAME "${IMPLIB_NAME}d")
  endif()
endif()

add_sycl_rt_library(${LIB_NAME} sycl_object ${LIB_OUTPUT_NAME}
                    SOURCES ${LIBSYCL_SOURCES} IMPLIB_OUTPUT_NAME ${IMPLIB_NAME})

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
  add_library(sycl ALIAS ${LIB_NAME})
endif()

install(TARGETS ${LIBSYCL_RT_LIBS}
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT libsycl
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT libsycl
  RUNTIME DESTINATION "bin" COMPONENT libsycl)
