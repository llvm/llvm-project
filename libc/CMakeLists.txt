cmake_minimum_required(VERSION 3.20.0)

# Include LLVM's cmake policies.
if(NOT DEFINED LLVM_COMMON_CMAKE_UTILS)
  set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
endif()
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

if (LIBC_CMAKE_VERBOSE_LOGGING)
  get_directory_property(LIBC_OLD_PREPROCESSOR_DEFS COMPILE_DEFINITIONS)
  foreach(OLD_DEF ${LIBC_OLD_PREPROCESSOR_DEFS})
    message(STATUS "Undefining ${OLD_DEF}")
  endforeach()
endif()
set_directory_properties(PROPERTIES
  # `llvm-project/llvm/CMakeLists.txt` adds the following directive
  # `include_directories( ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR})` We
  # undo it to be able to precisely control what is getting included.
  INCLUDE_DIRECTORIES ""
  # `llvm/cmake/modules/HandleLLVMOptions.cmake` uses `add_compile_definitions`
  # to set a few preprocessor defines which we do not want.
  COMPILE_DEFINITIONS ""
)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions("-D_DEBUG")
endif()

# Default to C++17
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# The top-level source directory.
set(LIBC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# The top-level directory in which libc is being built.
set(LIBC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(LIBC_ENABLE_USE_BY_CLANG OFF CACHE BOOL "Whether or not to place libc in a build directory findable by a just built clang")

# Defining a global namespace to enclose all libc functions.
set(LIBC_NAMESPACE "__llvm_libc_${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}_${LLVM_VERSION_PATCH}_${LLVM_VERSION_SUFFIX}"
  CACHE STRING "The namespace to use to enclose internal implementations. Must start with '__llvm_libc'."
)

if(LLVM_LIBC_FULL_BUILD OR LLVM_LIBC_GPU_BUILD)
  if(NOT LIBC_HDRGEN_EXE)
    # We need to set up hdrgen first since other targets depend on it.
    add_subdirectory(utils/LibcTableGenUtil)
    add_subdirectory(utils/HdrGen)
    # Calling add_tablegen sets variables like LIBC_TABLEGEN_EXE in
    # PARENT_SCOPE which get lost until saved in the cache.
    set(LIBC_TABLEGEN_EXE "${LIBC_TABLEGEN_EXE}" CACHE INTERNAL "")
    set(LIBC_TABLEGEN_TARGET "${LIBC_TABLEGEN_TARGET}" CACHE INTERNAL "")
  else()
    message(STATUS "Will use ${LIBC_HDRGEN_EXE} for libc header generation.")
  endif()
endif()

set(NEED_LIBC_HDRGEN FALSE)
if(NOT LLVM_RUNTIMES_BUILD)
  if("libc" IN_LIST LLVM_ENABLE_RUNTIMES)
    set(NEED_LIBC_HDRGEN TRUE)
  else()
    foreach(_name ${LLVM_RUNTIME_TARGETS})
      if("libc" IN_LIST RUNTIMES_${_name}_LLVM_ENABLE_RUNTIMES)
        set(NEED_LIBC_HDRGEN TRUE)
        break()
      endif()
    endforeach()
  endif()
endif()
option(LIBC_HDRGEN_ONLY "Only build the 'libc-hdrgen' executable" OFF)
if(LIBC_HDRGEN_ONLY OR NEED_LIBC_HDRGEN)
  # When libc is build as part of the runtimes/bootstrap build's CMake run, we
  # only need to build the host tools to build the libc. So, we just do enough
  # to build libc-hdrgen and return.

  # Always make the RPC server availible to other projects for GPU mode.
  if(LLVM_LIBC_GPU_BUILD)
    add_subdirectory(utils/gpu/server)
  endif()
  return()
endif()
unset(NEED_LIBC_HDRGEN)

option(LIBC_CMAKE_VERBOSE_LOGGING
  "Log details warnings and notifications during CMake configuration." OFF)

# Path libc/scripts directory.
set(LIBC_BUILD_SCRIPTS_DIR "${LIBC_SOURCE_DIR}/utils/build_scripts")

if(NOT LIBC_NAMESPACE MATCHES "^__llvm_libc")
  message(FATAL_ERROR "Invalid LIBC_NAMESPACE. Must start with '__llvm_libc' was '${LIBC_NAMESPACE}'")
endif()

message(STATUS "Setting LIBC_NAMESPACE namespace to '${LIBC_NAMESPACE}'")
add_compile_definitions(LIBC_NAMESPACE=${LIBC_NAMESPACE})

# Flags to pass down to the compiler while building the libc functions.
set(LIBC_COMPILE_OPTIONS_DEFAULT "" CACHE STRING "Architecture to tell clang to optimize for (e.g. -march=... or -mcpu=...)")

list(APPEND LIBC_COMPILE_OPTIONS_DEFAULT ${LIBC_COMMON_TUNE_OPTIONS})

# Check --print-resource-dir to find the compiler resource dir if this flag
# is supported by the compiler.
execute_process(
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND ${CMAKE_CXX_COMPILER} --print-resource-dir
  RESULT_VARIABLE COMMAND_RETURN_CODE
  OUTPUT_VARIABLE COMPILER_RESOURCE_DIR
)
# Retrieve the host compiler's resource dir.
if(COMMAND_RETURN_CODE EQUAL 0)
  set(COMPILER_RESOURCE_DIR
    "${COMPILER_RESOURCE_DIR}" CACHE PATH "path to compiler resource dir"
  )
  message(STATUS "Set COMPILER_RESOURCE_DIR to "
                 "${COMPILER_RESOURCE_DIR} using --print-resource-dir")
else()
  if (LIBC_TARGET_OS_IS_GPU)
    message(FATAL_ERROR "COMPILER_RESOURCE_DIR must be set for GPU builds")
  else()
    set(COMPILER_RESOURCE_DIR OFF)
    message(STATUS "COMPILER_RESOURCE_DIR not set
                    --print-resource-dir not supported by host compiler")
  endif()
endif()

option(LLVM_LIBC_FULL_BUILD "Build and test LLVM libc as if it is the full libc" OFF)
option(LLVM_LIBC_IMPLEMENTATION_DEFINED_TEST_BEHAVIOR "Build LLVM libc tests assuming our implementation-defined behavior" ON)
option(LLVM_LIBC_ENABLE_LINTING "Enables linting of libc source files" OFF)

option(LIBC_GPU_BUILD "Build libc for the GPU. All CPU build options will be ignored." OFF)
set(LIBC_TARGET_TRIPLE "" CACHE STRING "The target triple for the libc build.")

option(LIBC_CONFIG_PATH "The path to user provided folder that configures the build for the target system." OFF)

set(LIBC_ENABLE_UNITTESTS ON)
set(LIBC_ENABLE_HERMETIC_TESTS ${LLVM_LIBC_FULL_BUILD})

# Defines LIBC_TARGET_ARCHITECTURE and associated macros.
include(LLVMLibCArchitectures)

set(LIBC_CONFIG_JSON_FILE_LIST "")

if(NOT LIBC_CONFIG_PATH)
  list(APPEND LIBC_CONFIG_JSON_FILE_LIST "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}")
  if(EXISTS "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_ARCHITECTURE}")
    list(APPEND LIBC_CONFIG_JSON_FILE_LIST "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_ARCHITECTURE}")
    set(LIBC_CONFIG_PATH "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_ARCHITECTURE}")
  elseif(EXISTS "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}")
    set(LIBC_CONFIG_PATH "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}")
  endif()
else()
  list(APPEND LIBC_CONFIG_JSON_FILE_LIST "${LIBC_CONFIG_PATH}")
endif()

if(NOT LIBC_CONFIG_PATH)
  message(FATAL_ERROR "Configs for the platform '${LIBC_TARGET_OS}/${LIBC_TARGET_ARCHITECTURE}' do not exist and LIBC_CONFIG_PATH is not set.")
elseif(LIBC_CMAKE_VERBOSE_LOGGING)
  message(STATUS "Path for config files is: ${LIBC_CONFIG_PATH}")
endif()

# option(LIBC_ENABLE_WIDE_CHARACTERS
# "Whether to enable wide character functions on supported platforms. This may
# also set flags to enable or disable wide character support within other
# functions (e.g. printf)." ON)

#TODO: Add carve-out specific config files to the list here.

include(LibcConfig)
# Config loading happens in three steps:
# 1. Load the config file config/config.json and set up config vars.
# 2. Load config/${LIBC_TARGET_OS}/config.json if available and override
#    vars as suitable.
# 3. Load config/${LIBC_TARGET_OS}/${LIBC_TARGET_ARCH}/config.json is
#    available and override vars as suitable.
# All the three steps will not override options already set from the
# CMake command line. That is, the CMake command line option values take
# precedence over the values in config.json files.
set(main_config_file ${LIBC_SOURCE_DIR}/config/config.json)
read_libc_config(${main_config_file} global_config)
foreach(opt IN LISTS global_config)
  string(JSON opt_name ERROR_VARIABLE json_error MEMBER ${opt} 0)
  if(json_error)
    message(FATAL_ERROR ${json_error})
  endif()
  if(DEFINED ${opt_name})
    # The option is already defined from the command line so we ignore it here.
    # We still make note of it so that further config load can also ignore
    # this option.
    message(STATUS "${opt_name}: ${${opt_name}} (from command line)")
    list(APPEND cmd_line_conf ${opt_name})
    continue()
  endif()

  string(JSON opt_object ERROR_VARIABLE json_error GET ${opt} ${opt_name})
  if(json_error)
    message(FATAL_ERROR "Error reading info of option '${opt_name}': ${json_error}")
  endif()
  string(JSON opt_value ERROR_VARIABLE json_error GET ${opt_object} "value")
  if(json_error)
    message(FATAL_ERROR ${json_error})
  endif()
  message(STATUS "${opt_name}: ${opt_value}")
  set(${opt_name} ${opt_value})
endforeach()
generate_config_doc(${main_config_file} ${LIBC_SOURCE_DIR}/docs/configure.rst)

# Load each target specific config.
foreach(config_path IN LISTS LIBC_CONFIG_JSON_FILE_LIST)
  if(LIBC_CMAKE_VERBOSE_LOGGING)
    message(STATUS "Loading additional config: '${config_path}/config.json'")
  endif()
  load_libc_config(${config_path}/config.json ${cmd_line_conf})
endforeach()

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND LIBC_ENABLE_USE_BY_CLANG)
  set(LIBC_INCLUDE_DIR ${LLVM_BINARY_DIR}/include/${LLVM_DEFAULT_TARGET_TRIPLE})
  set(LIBC_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${LLVM_DEFAULT_TARGET_TRIPLE})
  set(LIBC_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE})
else()
  if(NOT LIBC_ENABLE_USE_BY_CLANG)
    set(LIBC_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
    set(LIBC_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
  elseif(LLVM_LIBRARY_OUTPUT_INTDIR)
    set(LIBC_INCLUDE_DIR ${LLVM_BINARY_DIR}/include)
    set(LIBC_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  else()
    set(LIBC_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
    set(LIBC_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  endif()
  if(LIBC_TARGET_OS_IS_GPU)
    set(LIBC_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${LIBC_TARGET_TRIPLE})
  else()
    set(LIBC_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
  endif()
endif()

if(LIBC_TARGET_TRIPLE)
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LIBC_TARGET_TRIPLE})
elseif(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT LIBC_GPU_BUILD)
  set(LIBC_INSTALL_LIBRARY_DIR
      lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE})
else()
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX})
endif()

if(LIBC_TARGET_OS_IS_GPU)
  include(prepare_libc_gpu_build)
  set(LIBC_ENABLE_UNITTESTS OFF)
endif()

include(LLVMLibCCheckMPFR)

if(LLVM_LIBC_CLANG_TIDY)
  set(LLVM_LIBC_ENABLE_LINTING ON)
endif()

if(LLVM_LIBC_ENABLE_LINTING)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(LLVM_LIBC_ENABLE_LINTING OFF)
    message(WARNING "C++ compiler is not clang++, linting with be disabled.")
  else()
    if (NOT LLVM_LIBC_CLANG_TIDY)
      find_program(LLVM_LIBC_CLANG_TIDY NAMES clang-tidy)
    endif()

    if(LLVM_LIBC_CLANG_TIDY)
      # Check clang-tidy major version.
      execute_process(COMMAND ${LLVM_LIBC_CLANG_TIDY} "--version"
        OUTPUT_VARIABLE CLANG_TIDY_OUTPUT
        ERROR_VARIABLE CLANG_TIDY_ERROR
        RESULT_VARIABLE CLANG_TIDY_RESULT)

      if(CLANG_TIDY_RESULT AND NOT CLANG_TIDY_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to execute '${LLVM_LIBC_CLANG_TIDY} --version'
          output : '${CLANG_TIDY_OUTPUT}'
          error  : '${CLANG_TIDY_ERROR}'
          result : '${CLANG_TIDY_RESULT}'
          ")
      endif()
      string(REGEX MATCH "[0-9]+" CLANG_TIDY_VERSION "${CLANG_TIDY_OUTPUT}")
      string(REGEX MATCH "[0-9]+" CLANG_MAJOR_VERSION
        "${CMAKE_CXX_COMPILER_VERSION}")

      if(NOT CLANG_TIDY_VERSION EQUAL CLANG_MAJOR_VERSION)
        set(LLVM_LIBC_ENABLE_LINTING OFF)
        message(WARNING "
          'clang-tidy' (version ${CLANG_TIDY_VERSION}) is not the same as
          'clang' (version ${CLANG_MAJOR_VERSION}).  Linting will
          be disabled.

          The path to the clang-tidy binary can be set manually by passing
          -DLLVM_LIBC_CLANG_TIDY=<path/to/clang-tidy> to CMake.")
      endif()
      add_custom_target(libc-lint)
    else()
      message(FATAL_ERROR "
        Linting is enabled but 'clang-tidy' is not found!

        The path to the clang-tidy binary can be set manually by passing
        -DLLVM_LIBC_CLANG_TIDY=<path/to/clang-tidy> to CMake.

        To disable linting set LLVM_LIBC_ENABLE_LINTING to OFF
        (pass -DLLVM_LIBC_ENABLE_LINTING=OFF to cmake).")
    endif()
  endif()
endif()

option(LLVM_LIBC_INCLUDE_SCUDO "Include the SCUDO standalone as the allocator for LLVM libc" OFF)
if(LLVM_LIBC_INCLUDE_SCUDO)
  if (NOT ("compiler-rt" IN_LIST LLVM_ENABLE_PROJECTS OR "compiler-rt" IN_LIST LLVM_ENABLE_RUNTIMES))
    message(FATAL_ERROR "SCUDO cannot be included without adding compiler-rt to LLVM_ENABLE_PROJECTS or LLVM_ENABLE_RUNTIMES")
  endif()
endif()

option(LIBC_INCLUDE_DOCS "Build the libc documentation." ${LLVM_INCLUDE_DOCS})

include(CMakeParseArguments)
include(LLVMLibCCheckCpuFeatures)
include(CheckCompilerFeatures)
include(LLVMLibCRules)

set(TARGET_LLVMLIBC_ENTRYPOINTS "")
set(TARGET_LIBC_ENTRYPOINTS "")
set(TARGET_LIBM_ENTRYPOINTS "")
set(TARGET_LLVMLIBC_REMOVED_ENTRYPOINTS "")

# Check entrypoints.txt
if(EXISTS "${LIBC_CONFIG_PATH}/entrypoints.txt")
    include("${LIBC_CONFIG_PATH}/entrypoints.txt")
else()
  message(FATAL_ERROR "${LIBC_CONFIG_PATH}/entrypoints.txt file not found.")
endif()

# Check headers.txt
if(EXISTS "${LIBC_CONFIG_PATH}/headers.txt")
    include("${LIBC_CONFIG_PATH}/headers.txt")
elseif(LLVM_LIBC_FULL_BUILD)
  message(FATAL_ERROR "${LIBC_CONFIG_PATH}/headers.txt file not found and fullbuild requested.")
endif()

# Check exclude.txt that appends to LIBC_EXCLUDE_ENTRYPOINTS list
if(EXISTS "${LIBC_CONFIG_PATH}/exclude.txt")
    include("${LIBC_CONFIG_PATH}/exclude.txt")
endif()

# #TODO: Set up support for premade configs adding their own exclude lists.

foreach(removed_entrypoint IN LISTS TARGET_LLVMLIBC_REMOVED_ENTRYPOINTS)
  if(LIBC_CMAKE_VERBOSE_LOGGING)
    message(STATUS "Removing entrypoint ${removed_entrypoint}")
  endif()
  list(REMOVE_ITEM TARGET_LLVMLIBC_ENTRYPOINTS ${removed_entrypoint})
  list(REMOVE_ITEM TARGET_LIBC_ENTRYPOINTS ${removed_entrypoint})
  list(REMOVE_ITEM TARGET_LIBM_ENTRYPOINTS ${removed_entrypoint})
endforeach()

set(TARGET_ENTRYPOINT_NAME_LIST "")
foreach(entrypoint IN LISTS TARGET_LLVMLIBC_ENTRYPOINTS)
  string(FIND ${entrypoint} "." last_dot_loc REVERSE)
  if(${last_dot_loc} EQUAL -1)
    message(FATAL_ERROR "Invalid entrypoint target name ${entrypoint}; Expected"
                        " a '.' (dot) in the name.")
  endif()
  math(EXPR name_loc "${last_dot_loc} + 1")
  string(SUBSTRING ${entrypoint} ${name_loc} -1 entrypoint_name)
  list(APPEND TARGET_ENTRYPOINT_NAME_LIST ${entrypoint_name})
endforeach()

add_subdirectory(include)
add_subdirectory(config)
add_subdirectory(src)
add_subdirectory(utils)

if(LLVM_LIBC_FULL_BUILD)
  # The startup system can potentially depend on the library components so add
  # it after the library implementation directories.
  add_subdirectory(startup)
endif()

# The lib and test directories are added at the very end as tests
# and libraries potentially draw from the components present in all
# of the other directories.
add_subdirectory(lib)
if(LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(fuzzing)
endif()

if(LIBC_INCLUDE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if (LIBC_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()
