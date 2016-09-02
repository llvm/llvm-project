macro(find_llvm_parts)
# Rely on llvm-config.
  set(CONFIG_OUTPUT)
  if(NOT LLVM_CONFIG_PATH)
    find_program(LLVM_CONFIG_PATH "llvm-config")
  endif()
  if(DEFINED LLVM_PATH)
    set(LLVM_INCLUDE_DIR ${LLVM_INCLUDE_DIR} CACHE PATH "Path to llvm/include")
    set(LLVM_PATH ${LLVM_PATH} CACHE PATH "Path to LLVM source tree")
    set(LLVM_MAIN_SRC_DIR ${LLVM_PATH})
    set(LLVM_CMAKE_PATH "${LLVM_PATH}/cmake/modules")
  elseif(LLVM_CONFIG_PATH)
    message(STATUS "Found LLVM_CONFIG_PATH as ${LLVM_CONFIG_PATH}")
    set(CONFIG_COMMAND ${LLVM_CONFIG_PATH}
      "--includedir"
      "--prefix"
      "--src-root")
    execute_process(
      COMMAND ${CONFIG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
    )
    if(NOT HAD_ERROR)
      string(REGEX REPLACE
        "[ \t]*[\r\n]+[ \t]*" ";"
        CONFIG_OUTPUT ${CONFIG_OUTPUT})
    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()

    list(GET CONFIG_OUTPUT 0 INCLUDE_DIR)
    list(GET CONFIG_OUTPUT 1 LLVM_OBJ_ROOT)
    list(GET CONFIG_OUTPUT 2 MAIN_SRC_DIR)

    set(LLVM_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
    set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
    set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
    set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")
  else()
    set(LLVM_FOUND OFF)
    return()
  endif()

  if (NOT EXISTS ${LLVM_MAIN_SRC_DIR})
    set(LLVM_FOUND OFF)
    message(WARNING "Not found: ${LLVM_MAIN_SRC_DIR}")
    return()
  endif()

  if(NOT EXISTS ${LLVM_CMAKE_PATH})
    set(LLVM_FOUND OFF)
    message(WARNING "Not found: ${LLVM_CMAKE_PATH}")
    return()
  endif()

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")

  set(LLVM_FOUND ON)
endmacro(find_llvm_parts)

# If this is a standalone build not running as an external project of LLVM
# we need to later make some decisions differently.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # The intent is that this doesn't necessarily mean the LLVM is installed (it
  # could be a build directory), but it means we need to treat the LLVM
  # directory as read-only.
  set(LIBCXX_USING_INSTALLED_LLVM 1)
endif()

if (LIBCXX_USING_INSTALLED_LLVM OR LIBCXX_STANDALONE_BUILD)
  set(LIBCXX_STANDALONE_BUILD 1)
  message(STATUS "Configuring for standalone build.")

  find_llvm_parts()

  # LLVM Options --------------------------------------------------------------
  include(FindPythonInterp)
  if( NOT PYTHONINTERP_FOUND )
    message(WARNING "Failed to find python interpreter. "
                    "The libc++ test suite will be disabled.")
    set(LLVM_INCLUDE_TESTS OFF)
  endif()

  if (NOT DEFINED LLVM_INCLUDE_TESTS)
    set(LLVM_INCLUDE_TESTS ${LLVM_FOUND})
  endif()
  if (NOT DEFINED LLVM_INCLUDE_DOCS)
    set(LLVM_INCLUDE_DOCS ${LLVM_FOUND})
  endif()
  if (NOT DEFINED LLVM_ENABLE_SPHINX)
    set(LLVM_ENABLE_SPHINX OFF)
  endif()

  # Required LIT Configuration ------------------------------------------------
  # Define the default arguments to use with 'lit', and an option for the user
  # to override.
  set(LIT_ARGS_DEFAULT "-sv --show-xfail --show-unsupported")
  if (MSVC OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

  # Required doc configuration
  if (LLVM_ENABLE_SPHINX)
    message(STATUS "Sphinx enabled.")
    find_package(Sphinx REQUIRED)
  else()
    message(STATUS "Sphinx disabled.")
  endif()

  # FIXME - This is cribbed from HandleLLVMOptions.cmake.
  if(WIN32)
    set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
    if(CYGWIN)
      set(LLVM_ON_WIN32 0)
      set(LLVM_ON_UNIX 1)
    else(CYGWIN)
      set(LLVM_ON_WIN32 1)
      set(LLVM_ON_UNIX 0)
    endif(CYGWIN)
  else(WIN32)
    if(UNIX)
      set(LLVM_ON_WIN32 0)
      set(LLVM_ON_UNIX 1)
      if(APPLE)
        set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
      else(APPLE)
        set(LLVM_HAVE_LINK_VERSION_SCRIPT 1)
      endif(APPLE)
    else(UNIX)
      MESSAGE(SEND_ERROR "Unable to determine platform")
    endif(UNIX)
  endif(WIN32)

  # Add LLVM Functions --------------------------------------------------------
  include(AddLLVM OPTIONAL)
endif()
