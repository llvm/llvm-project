# -*- Mode: CMakeLists.txt; -*-
#
# See LICENSE.txt in top-level directory.
#

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Check if the built-in Argobots needs to be used
set(LIBOMP_USE_ARGOBOTS FALSE CACHE BOOL
  "Use Argobots (http://www.argobots.org) as threading model?")
if(${LIBOMP_USE_ARGOBOTS} AND (NOT LIBOMP_ARGOBOTS_INSTALL_DIR))
  set(LIBOMP_USE_BUILTIN_ARGOBOTS TRUE CACHE BOOL "")
else()
  set(LIBOMP_USE_BUILTIN_ARGOBOTS FALSE CACHE BOOL "")
endif()

if(${LIBOMP_USE_BUILTIN_ARGOBOTS})
  # Check if the built-in Argobots exists.
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/argobots/configure.ac")
    if (EXISTS "${PROJECT_SOURCE_DIR}/.git")
      message(STATUS "Running `git submodule update --init`")
      execute_process(COMMAND git submodule update --init
                      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                      RESULT_VARIABLE GIT_SUBMOD_RESULT)
      if (NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(STATUS "git submodule update --init failed with `${GIT_SUBMOD_RESULT}`. ")
      else()
        message(STATUS "Running `git submodule update --init` - Success")
      endif()
    endif()
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/argobots/configure.ac")
        message(FATAL_ERROR "The built-in Argobots is not found. "
                            "Please run `git submodule update --init` if you get BOLT via GitHub, or "
                            "download Argobots manually and place it at `external/argobots`.")
    endif()
  endif()
  # Use the built-in Argobots
  set(ABT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/argobots)
  set(ABT_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/argobots)
  set(ABT_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/argobots/install)
  set(ABT_AUTOGEN ${ABT_SRC_DIR}/autogen.sh)
  set(ABT_CONFIGURE ${ABT_SRC_DIR}/configure)
  if ((NOT EXISTS ${ABT_CONFIGURE}) OR (${ABT_AUTOGEN} IS_NEWER_THAN ${ABT_CONFIGURE}))
    execute_process(
      COMMAND ./autogen.sh
      WORKING_DIRECTORY ${ABT_SRC_DIR}
    )
  endif()

  include(ExternalProject)
  ExternalProject_Add(libabt
    SOURCE_DIR ${ABT_SRC_DIR}
    BINARY_DIR ${ABT_BUILD_DIR}
    CONFIGURE_COMMAND ${ABT_CONFIGURE} --prefix=${ABT_INSTALL_DIR} CC=${CMAKE_C_COMPILER}
  )
  # Install libraries.
  if(${OPENMP_STANDALONE_BUILD})
    set(ABT_LIBS_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib)
  else()
    set(ABT_LIBS_INSTALL_PATH ${OPENMP_INSTALL_LIBDIR})
  endif()
  # pkgconfig is not installed.
  install(DIRECTORY ${ABT_INSTALL_DIR}/lib/
          DESTINATION ${ABT_LIBS_INSTALL_PATH}
          PATTERN pkgconfig EXCLUDE)
  # Install headers.  pkgconfig is not installed.
  if(${OPENMP_STANDALONE_BUILD})
    set(ABT_HEADERS_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/include)
  else()
    string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION ${PACKAGE_VERSION})
    set(ABT_HEADERS_INSTALL_PATH "${OPENMP_INSTALL_LIBDIR}/clang/${CLANG_VERSION}/include")
  endif()
  install(FILES ${ABT_INSTALL_DIR}/include/abt.h
          DESTINATION ${ABT_HEADERS_INSTALL_PATH})

  set(LIBOMP_ARGOBOTS_INSTALL_DIR ${ABT_INSTALL_DIR} PARENT_SCOPE)
else()
  set(LIBOMP_ARGOBOTS_INSTALL_DIR /usr/local CACHE PATH
    "Install path for Argobots")
endif()

if(${LIBOMP_USE_ARGOBOTS})
  # Need to be python-bool.
  set(LIBOMP_TEST_USE_ARGOBOTS True PARENT_SCOPE)
else()
  set(LIBOMP_TEST_USE_ARGOBOTS False PARENT_SCOPE)
endif()
