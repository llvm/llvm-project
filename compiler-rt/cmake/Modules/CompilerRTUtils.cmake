include(CMakePushCheckState)
include(CheckSymbolExists)

# Because compiler-rt spends a lot of time setting up custom compile flags,
# define a handy helper function for it. The compile flags setting in CMake
# has serious issues that make its syntax challenging at best.
function(set_target_compile_flags target)
  set(argstring "")
  foreach(arg ${ARGN})
    set(argstring "${argstring} ${arg}")
  endforeach()
  set_property(TARGET ${target} PROPERTY COMPILE_FLAGS "${argstring}")
endfunction()

function(set_target_link_flags target)
  set(argstring "")
  foreach(arg ${ARGN})
    set(argstring "${argstring} ${arg}")
  endforeach()
  set_property(TARGET ${target} PROPERTY LINK_FLAGS "${argstring}")
endfunction()

# Set the variable var_PYBOOL to True if var holds a true-ish string,
# otherwise set it to False.
macro(pythonize_bool var)
  if (${var})
    set(${var}_PYBOOL True)
  else()
    set(${var}_PYBOOL False)
  endif()
endmacro()

# Appends value to all lists in ARGN, if the condition is true.
macro(append_list_if condition value)
  if(${condition})
    foreach(list ${ARGN})
      list(APPEND ${list} ${value})
    endforeach()
  endif()
endmacro()

# Appends value to all strings in ARGN, if the condition is true.
macro(append_string_if condition value)
  if(${condition})
    foreach(str ${ARGN})
      set(${str} "${${str}} ${value}")
    endforeach()
  endif()
endmacro()

macro(append_rtti_flag polarity list)
  if(${polarity})
    append_list_if(COMPILER_RT_HAS_FRTTI_FLAG -frtti ${list})
    append_list_if(COMPILER_RT_HAS_GR_FLAG /GR ${list})
  else()
    append_list_if(COMPILER_RT_HAS_FNO_RTTI_FLAG -fno-rtti ${list})
    append_list_if(COMPILER_RT_HAS_GR_FLAG /GR- ${list})
  endif()
endmacro()

macro(list_intersect output input1 input2)
  set(${output})
  foreach(it ${${input1}})
    list(FIND ${input2} ${it} index)
    if( NOT (index EQUAL -1))
      list(APPEND ${output} ${it})
    endif()
  endforeach()
endmacro()

function(list_replace input_list old new)
  set(replaced_list)
  foreach(item ${${input_list}})
    if(${item} STREQUAL ${old})
      list(APPEND replaced_list ${new})
    else()
      list(APPEND replaced_list ${item})
    endif()
  endforeach()
  set(${input_list} "${replaced_list}" PARENT_SCOPE)
endfunction()

# Takes ${ARGN} and puts only supported architectures in @out_var list.
function(filter_available_targets out_var)
  set(archs ${${out_var}})
  foreach(arch ${ARGN})
    list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
    if(NOT (ARCH_INDEX EQUAL -1) AND CAN_TARGET_${arch})
      list(APPEND archs ${arch})
    endif()
  endforeach()
  set(${out_var} ${archs} PARENT_SCOPE)
endfunction()

# Add $arch as supported with no additional flags.
macro(add_default_target_arch arch)
  set(TARGET_${arch}_CFLAGS "")
  set(CAN_TARGET_${arch} 1)
  list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
endmacro()

function(check_compile_definition def argstring out_var)
  if("${def}" STREQUAL "")
    set(${out_var} TRUE PARENT_SCOPE)
    return()
  endif()
  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${argstring}")
  check_symbol_exists(${def} "" ${out_var})
  cmake_pop_check_state()
endfunction()

# test_target_arch(<arch> <def> <target flags...>)
# Checks if architecture is supported: runs host compiler with provided
# flags to verify that:
#   1) <def> is defined (if non-empty)
#   2) simple file can be successfully built.
# If successful, saves target flags for this architecture.
macro(test_target_arch arch def)
  set(TARGET_${arch}_CFLAGS ${ARGN})
  set(TARGET_${arch}_LINK_FLAGS ${ARGN})
  set(argstring "")
  foreach(arg ${ARGN})
    set(argstring "${argstring} ${arg}")
  endforeach()
  check_compile_definition("${def}" "${argstring}" HAS_${arch}_DEF)
  if(NOT DEFINED CAN_TARGET_${arch})
    if(NOT HAS_${arch}_DEF)
      set(CAN_TARGET_${arch} FALSE)
    elseif(TEST_COMPILE_ONLY)
      try_compile_only(CAN_TARGET_${arch} FLAGS ${TARGET_${arch}_CFLAGS})
    else()
      set(FLAG_NO_EXCEPTIONS "")
      if(COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG)
        set(FLAG_NO_EXCEPTIONS " -fno-exceptions ")
      endif()
      set(SAVED_CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${argstring}")
      try_compile(CAN_TARGET_${arch} ${CMAKE_BINARY_DIR} ${SIMPLE_SOURCE}
                  COMPILE_DEFINITIONS "${TARGET_${arch}_CFLAGS} ${FLAG_NO_EXCEPTIONS}"
                  OUTPUT_VARIABLE TARGET_${arch}_OUTPUT)
      set(CMAKE_EXE_LINKER_FLAGS ${SAVED_CMAKE_EXE_LINKER_FLAGS})
    endif()
  endif()
  if(${CAN_TARGET_${arch}})
    list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
  elseif("${COMPILER_RT_DEFAULT_TARGET_ARCH}" STREQUAL "${arch}" AND
         COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE)
    # Bail out if we cannot target the architecture we plan to test.
    message(FATAL_ERROR "Cannot compile for ${arch}:\n${TARGET_${arch}_OUTPUT}")
  endif()
endmacro()

macro(detect_target_arch)
  check_symbol_exists(__arm__ "" __ARM)
  check_symbol_exists(__aarch64__ "" __AARCH64)
  check_symbol_exists(__x86_64__ "" __X86_64)
  check_symbol_exists(__i386__ "" __I386)
  check_symbol_exists(__mips__ "" __MIPS)
  check_symbol_exists(__mips64__ "" __MIPS64)
  check_symbol_exists(__powerpc__ "" __PPC)
  check_symbol_exists(__powerpc64__ "" __PPC64)
  check_symbol_exists(__powerpc64le__ "" __PPC64LE)
  check_symbol_exists(__riscv "" __RISCV)
  check_symbol_exists(__s390x__ "" __S390X)
  check_symbol_exists(__sparc "" __SPARC)
  check_symbol_exists(__sparcv9 "" __SPARCV9)
  check_symbol_exists(__wasm32__ "" __WEBASSEMBLY32)
  check_symbol_exists(__wasm64__ "" __WEBASSEMBLY64)
  check_symbol_exists(__ve__ "" __VE)
  if(__ARM)
    add_default_target_arch(arm)
  elseif(__AARCH64)
    add_default_target_arch(aarch64)
  elseif(__X86_64)
    add_default_target_arch(x86_64)
  elseif(__I386)
    add_default_target_arch(i386)
  elseif(__MIPS64) # must be checked before __MIPS
    add_default_target_arch(mips64)
  elseif(__MIPS)
    add_default_target_arch(mips)
  elseif(__PPC64) # must be checked before __PPC
    add_default_target_arch(powerpc64)
  elseif(__PPC64LE)
    add_default_target_arch(powerpc64le)
  elseif(__PPC)
    add_default_target_arch(powerpc)
  elseif(__RISCV)
    if(CMAKE_SIZEOF_VOID_P EQUAL "4")
      add_default_target_arch(riscv32)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL "8")
      add_default_target_arch(riscv64)
    else()
      message(FATAL_ERROR "Unsupport XLEN for RISC-V")
    endif()
  elseif(__S390X)
    add_default_target_arch(s390x)
  elseif(__SPARCV9)
    add_default_target_arch(sparcv9)
  elseif(__SPARC)
    add_default_target_arch(sparc)
  elseif(__WEBASSEMBLY32)
    add_default_target_arch(wasm32)
  elseif(__WEBASSEMBLY64)
    add_default_target_arch(wasm64)
  elseif(__VE)
    add_default_target_arch(ve)
  endif()
endmacro()

function(get_compiler_rt_root_source_dir ROOT_DIR_VAR)
  # Compute the path to the root of the Compiler-RT source tree
  # regardless of how the project was configured.
  #
  # This function is useful because using `${CMAKE_SOURCE_DIR}`
  # is error prone due to the numerous ways Compiler-RT can be
  # configured.
  #
  # `ROOT_DIR_VAR` - the name of the variable to write the result to.
  #
  # TODO(dliew): When CMake min version is 3.17 or newer use
  # `CMAKE_CURRENT_FUNCTION_LIST_DIR` instead.
  if ("${ROOT_DIR_VAR}" STREQUAL "")
    message(FATAL_ERROR "ROOT_DIR_VAR cannot be empty")
  endif()

  # Compiler-rt supports different source root paths.
  # Handle each case here.
  set(PATH_TO_COMPILER_RT_SOURCE_ROOT "")
  if (DEFINED CompilerRTBuiltins_SOURCE_DIR)
    # Compiler-RT Builtins standalone build.
    # `llvm-project/compiler-rt/lib/builtins`
    set(PATH_TO_COMPILER_RT_SOURCE_ROOT "${CompilerRTBuiltins_SOURCE_DIR}/../../")
  elseif(DEFINED CompilerRT_SOURCE_DIR)
    # Compiler-RT standalone build.
    # `llvm-project/compiler-rt`
    set(PATH_TO_COMPILER_RT_SOURCE_ROOT "${CompilerRT_SOURCE_DIR}")
  elseif (EXISTS "${CMAKE_SOURCE_DIR}/../compiler-rt")
    # In tree build with LLVM as the root project.
    # See `llvm-project/projects/`.
    # Assumes monorepo layout.
    set(PATH_TO_COMPILER_RT_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/../compiler-rt")
  else()
    message(FATAL_ERROR "Unhandled Compiler-RT source root configuration.")
  endif()

  get_filename_component(ROOT_DIR "${PATH_TO_COMPILER_RT_SOURCE_ROOT}" ABSOLUTE)
  if (NOT EXISTS "${ROOT_DIR}")
    message(FATAL_ERROR "Path \"${ROOT_DIR}\" doesn't exist")
  endif()

  # Sanity check: Make sure we can locate the current source file via the
  # computed path.
  set(PATH_TO_CURRENT_FILE "${ROOT_DIR}/cmake/Modules/CompilerRTUtils.cmake")
  if (NOT EXISTS "${PATH_TO_CURRENT_FILE}")
    message(FATAL_ERROR "Could not find \"${PATH_TO_CURRENT_FILE}\"")
  endif()

  set("${ROOT_DIR_VAR}" "${ROOT_DIR}" PARENT_SCOPE)
endfunction()

macro(load_llvm_config)
  if (NOT LLVM_CONFIG_PATH)
    find_program(LLVM_CONFIG_PATH "llvm-config"
                 DOC "Path to llvm-config binary")
    if (NOT LLVM_CONFIG_PATH)
      message(WARNING "UNSUPPORTED COMPILER-RT CONFIGURATION DETECTED: "
                      "llvm-config not found.\n"
                      "Reconfigure with -DLLVM_CONFIG_PATH=path/to/llvm-config.")
    endif()
  endif()

  # Compute path to LLVM sources assuming the monorepo layout.
  # We don't set `LLVM_MAIN_SRC_DIR` directly to avoid overriding a user provided
  # CMake cache value.
  get_compiler_rt_root_source_dir(COMPILER_RT_ROOT_SRC_PATH)
  get_filename_component(LLVM_MAIN_SRC_DIR_DEFAULT "${COMPILER_RT_ROOT_SRC_PATH}/../llvm" ABSOLUTE)
  if (NOT EXISTS "${LLVM_MAIN_SRC_DIR_DEFAULT}")
    # TODO(dliew): Remove this legacy fallback path.
    message(WARNING
      "LLVM source tree not found at \"${LLVM_MAIN_SRC_DIR_DEFAULT}\". "
      "You are not using the monorepo layout. This configuration is DEPRECATED.")
  endif()

  set(FOUND_LLVM_CMAKE_PATH FALSE)
  if (LLVM_CONFIG_PATH)
    execute_process(
      COMMAND ${LLVM_CONFIG_PATH} "--obj-root" "--bindir" "--libdir" "--src-root" "--includedir"
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT)
    if (HAD_ERROR)
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()
    string(REGEX REPLACE "[ \t]*[\r\n]+[ \t]*" ";" CONFIG_OUTPUT ${CONFIG_OUTPUT})
    list(GET CONFIG_OUTPUT 0 BINARY_DIR)
    list(GET CONFIG_OUTPUT 1 TOOLS_BINARY_DIR)
    list(GET CONFIG_OUTPUT 2 LIBRARY_DIR)
    list(GET CONFIG_OUTPUT 3 MAIN_SRC_DIR)
    list(GET CONFIG_OUTPUT 4 INCLUDE_DIR)

    set(LLVM_BINARY_DIR ${BINARY_DIR} CACHE PATH "Path to LLVM build tree")
    set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
    set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
    set(LLVM_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Paths to LLVM headers")

    if (NOT EXISTS "${LLVM_MAIN_SRC_DIR_DEFAULT}")
      # TODO(dliew): Remove this legacy fallback path.
      message(WARNING
        "Consulting llvm-config for the LLVM source path "
        "as a fallback. This behavior will be removed in the future.")
      # We don't set `LLVM_MAIN_SRC_DIR` directly to avoid overriding a user
      # provided CMake cache value.
      set(LLVM_MAIN_SRC_DIR_DEFAULT "${MAIN_SRC_DIR}")
      message(STATUS "Using LLVM source path (${LLVM_MAIN_SRC_DIR_DEFAULT}) from llvm-config")
    endif()

    # Detect if we have the LLVMXRay and TestingSupport library installed and
    # available from llvm-config.
    execute_process(
      COMMAND ${LLVM_CONFIG_PATH} "--ldflags" "--libs" "xray"
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
      ERROR_QUIET)
    if (HAD_ERROR)
      message(WARNING "llvm-config finding xray failed with status ${HAD_ERROR}")
      set(COMPILER_RT_HAS_LLVMXRAY FALSE)
    else()
      string(REGEX REPLACE "[ \t]*[\r\n]+[ \t]*" ";" CONFIG_OUTPUT ${CONFIG_OUTPUT})
      list(GET CONFIG_OUTPUT 0 LDFLAGS)
      list(GET CONFIG_OUTPUT 1 LIBLIST)
      file(TO_CMAKE_PATH "${LDFLAGS}" LDFLAGS)
      file(TO_CMAKE_PATH "${LIBLIST}" LIBLIST)
      set(LLVM_XRAY_LDFLAGS ${LDFLAGS} CACHE STRING "Linker flags for LLVMXRay library")
      set(LLVM_XRAY_LIBLIST ${LIBLIST} CACHE STRING "Library list for LLVMXRay")
      set(COMPILER_RT_HAS_LLVMXRAY TRUE)
    endif()

    set(COMPILER_RT_HAS_LLVMTESTINGSUPPORT FALSE)
    execute_process(
      COMMAND ${LLVM_CONFIG_PATH} "--ldflags" "--libs" "testingsupport"
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
      ERROR_QUIET)
    if (HAD_ERROR)
      message(WARNING "llvm-config finding testingsupport failed with status ${HAD_ERROR}")
    elseif(COMPILER_RT_INCLUDE_TESTS)
      string(REGEX REPLACE "[ \t]*[\r\n]+[ \t]*" ";" CONFIG_OUTPUT ${CONFIG_OUTPUT})
      list(GET CONFIG_OUTPUT 0 LDFLAGS)
      list(GET CONFIG_OUTPUT 1 LIBLIST)
      if (LIBLIST STREQUAL "")
        message(WARNING "testingsupport library not installed, some tests will be skipped")
      else()
        file(TO_CMAKE_PATH "${LDFLAGS}" LDFLAGS)
        file(TO_CMAKE_PATH "${LIBLIST}" LIBLIST)
        set(LLVM_TESTINGSUPPORT_LDFLAGS ${LDFLAGS} CACHE STRING "Linker flags for LLVMTestingSupport library")
        set(LLVM_TESTINGSUPPORT_LIBLIST ${LIBLIST} CACHE STRING "Library list for LLVMTestingSupport")
        set(COMPILER_RT_HAS_LLVMTESTINGSUPPORT TRUE)
      endif()
    endif()

    # Make use of LLVM CMake modules.
    # --cmakedir is supported since llvm r291218 (4.0 release)
    execute_process(
      COMMAND ${LLVM_CONFIG_PATH} --cmakedir
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT)
    if(NOT HAD_ERROR)
      string(STRIP "${CONFIG_OUTPUT}" LLVM_CMAKE_PATH_FROM_LLVM_CONFIG)
      file(TO_CMAKE_PATH ${LLVM_CMAKE_PATH_FROM_LLVM_CONFIG} LLVM_CMAKE_PATH)
    else()
      file(TO_CMAKE_PATH ${LLVM_BINARY_DIR} LLVM_BINARY_DIR_CMAKE_STYLE)
      set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR_CMAKE_STYLE}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")
    endif()

    if (EXISTS "${LLVM_CMAKE_PATH}")
      list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
      # Get some LLVM variables from LLVMConfig.
      include("${LLVM_CMAKE_PATH}/LLVMConfig.cmake")
      set(FOUND_LLVM_CMAKE_PATH TRUE)
    else()
      set(FOUND_LLVM_CMAKE_PATH FALSE)
      message(WARNING "LLVM CMake path (${LLVM_CMAKE_PATH}) reported by llvm-config does not exist")
    endif()

    set(LLVM_LIBRARY_OUTPUT_INTDIR
      ${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
  endif()

  # Finally set the cache variable now that `llvm-config` has also had a chance
  # to set `LLVM_MAIN_SRC_DIR_DEFAULT`.
  set(LLVM_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR_DEFAULT}" CACHE PATH "Path to LLVM source tree")
  message(STATUS "LLVM_MAIN_SRC_DIR: \"${LLVM_MAIN_SRC_DIR}\"")
  if (NOT EXISTS "${LLVM_MAIN_SRC_DIR}")
    # TODO(dliew): Make this a hard error
    message(WARNING "LLVM_MAIN_SRC_DIR (${LLVM_MAIN_SRC_DIR}) does not exist. "
                    "You can override the inferred path by adding "
                    "`-DLLVM_MAIN_SRC_DIR=<path_to_llvm_src>` to your CMake invocation "
                    "where `<path_to_llvm_src>` is the path to the `llvm` directory in "
                    "the `llvm-project` repo. "
                    "This will be treated as error in the future.")
  endif()

  if (NOT FOUND_LLVM_CMAKE_PATH)
    # This configuration tries to configure without the prescence of `LLVMConfig.cmake`. It is
    # intended for testing purposes (generating the lit test suites) and will likely not support
    # a build of the runtimes in compiler-rt.
    include(CompilerRTMockLLVMCMakeConfig)
    compiler_rt_mock_llvm_cmake_config()
  endif()

endmacro()

macro(construct_compiler_rt_default_triple)
  if(COMPILER_RT_DEFAULT_TARGET_ONLY)
    if(DEFINED COMPILER_RT_DEFAULT_TARGET_TRIPLE)
      message(FATAL_ERROR "COMPILER_RT_DEFAULT_TARGET_TRIPLE isn't supported when building for default target only")
    endif()
    set(COMPILER_RT_DEFAULT_TARGET_TRIPLE ${CMAKE_C_COMPILER_TARGET})
  else()
    set(COMPILER_RT_DEFAULT_TARGET_TRIPLE ${TARGET_TRIPLE} CACHE STRING
          "Default triple for which compiler-rt runtimes will be built.")
  endif()

  if(DEFINED COMPILER_RT_TEST_TARGET_TRIPLE)
    # Backwards compatibility: this variable used to be called
    # COMPILER_RT_TEST_TARGET_TRIPLE.
    set(COMPILER_RT_DEFAULT_TARGET_TRIPLE ${COMPILER_RT_TEST_TARGET_TRIPLE})
  endif()

  string(REPLACE "-" ";" TARGET_TRIPLE_LIST ${COMPILER_RT_DEFAULT_TARGET_TRIPLE})
  list(GET TARGET_TRIPLE_LIST 0 COMPILER_RT_DEFAULT_TARGET_ARCH)

  # Map various forms of the architecture names to the canonical forms
  # (as they are used by clang, see getArchNameForCompilerRTLib).
  if("${COMPILER_RT_DEFAULT_TARGET_ARCH}" MATCHES "^i.86$")
    # Android uses i686, but that's remapped at a later stage.
    set(COMPILER_RT_DEFAULT_TARGET_ARCH "i386")
  endif()

  # Determine if test target triple is specified explicitly, and doesn't match the
  # default.
  if(NOT COMPILER_RT_DEFAULT_TARGET_TRIPLE STREQUAL TARGET_TRIPLE)
    set(COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE TRUE)
  else()
    set(COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE FALSE)
  endif()
endmacro()

# Filter out generic versions of routines that are re-implemented in an
# architecture specific manner. This prevents multiple definitions of the same
# symbols, making the symbol selection non-deterministic.
#
# We follow the convention that a source file that exists in a sub-directory
# (e.g. `ppc/divtc3.c`) is architecture-specific and that if a generic
# implementation exists it will be a top-level source file with the same name
# modulo the file extension (e.g. `divtc3.c`).
function(filter_builtin_sources inout_var name)
  set(intermediate ${${inout_var}})
  foreach(_file ${intermediate})
    get_filename_component(_file_dir ${_file} DIRECTORY)
    if (NOT "${_file_dir}" STREQUAL "")
      # Architecture specific file. If a generic version exists, print a notice
      # and ensure that it is removed from the file list.
      get_filename_component(_name ${_file} NAME)
      string(REGEX REPLACE "\\.S$" ".c" _cname "${_name}")
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_cname}")
        message(STATUS "For ${name} builtins preferring ${_file} to ${_cname}")
        list(REMOVE_ITEM intermediate ${_cname})
      endif()
    endif()
  endforeach()
  set(${inout_var} ${intermediate} PARENT_SCOPE)
endfunction()

function(get_compiler_rt_target arch variable)
  string(FIND ${COMPILER_RT_DEFAULT_TARGET_TRIPLE} "-" dash_index)
  string(SUBSTRING ${COMPILER_RT_DEFAULT_TARGET_TRIPLE} ${dash_index} -1 triple_suffix)
  if(COMPILER_RT_DEFAULT_TARGET_ONLY)
    # Use exact spelling when building only for the target specified to CMake.
    set(target "${COMPILER_RT_DEFAULT_TARGET_TRIPLE}")
  elseif(ANDROID AND ${arch} STREQUAL "i386")
    set(target "i686${triple_suffix}")
  else()
    set(target "${arch}${triple_suffix}")
  endif()
  set(${variable} ${target} PARENT_SCOPE)
endfunction()

function(get_compiler_rt_install_dir arch install_dir)
  if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
    get_compiler_rt_target(${arch} target)
    set(${install_dir} ${COMPILER_RT_INSTALL_PATH}/lib/${target} PARENT_SCOPE)
  else()
    set(${install_dir} ${COMPILER_RT_LIBRARY_INSTALL_DIR} PARENT_SCOPE)
  endif()
endfunction()

function(get_compiler_rt_output_dir arch output_dir)
  if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
    get_compiler_rt_target(${arch} target)
    set(${output_dir} ${COMPILER_RT_OUTPUT_DIR}/lib/${target} PARENT_SCOPE)
  else()
    set(${output_dir} ${COMPILER_RT_LIBRARY_OUTPUT_DIR} PARENT_SCOPE)
  endif()
endfunction()

# compiler_rt_process_sources(
#   <OUTPUT_VAR>
#   <SOURCE_FILE> ...
#  [ADDITIONAL_HEADERS <header> ...]
# )
#
# Process the provided sources and write the list of new sources
# into `<OUTPUT_VAR>`.
#
# ADDITIONAL_HEADERS     - Adds the supplied header to list of sources for IDEs.
#
# This function is very similar to `llvm_process_sources()` but exists here
# because we need to support standalone builds of compiler-rt.
function(compiler_rt_process_sources OUTPUT_VAR)
  cmake_parse_arguments(
    ARG
    ""
    ""
    "ADDITIONAL_HEADERS"
    ${ARGN}
  )
  set(sources ${ARG_UNPARSED_ARGUMENTS})
  set(headers "")
  if (XCODE OR MSVC_IDE OR CMAKE_EXTRA_GENERATOR)
    # For IDEs we need to tell CMake about header files.
    # Otherwise they won't show up in UI.
    set(headers ${ARG_ADDITIONAL_HEADERS})
    list(LENGTH headers headers_length)
    if (${headers_length} GREATER 0)
      set_source_files_properties(${headers}
        PROPERTIES HEADER_FILE_ONLY ON)
    endif()
  endif()
  set("${OUTPUT_VAR}" ${sources} ${headers} PARENT_SCOPE)
endfunction()

# Create install targets for a library and its parent component (if specified).
function(add_compiler_rt_install_targets name)
  cmake_parse_arguments(ARG "" "PARENT_TARGET" "" ${ARGN})

  if(ARG_PARENT_TARGET AND NOT TARGET install-${ARG_PARENT_TARGET})
    # The parent install target specifies the parent component to scrape up
    # anything not installed by the individual install targets, and to handle
    # installation when running the multi-configuration generators.
    add_custom_target(install-${ARG_PARENT_TARGET}
                      DEPENDS ${ARG_PARENT_TARGET}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=${ARG_PARENT_TARGET}
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    add_custom_target(install-${ARG_PARENT_TARGET}-stripped
                      DEPENDS ${ARG_PARENT_TARGET}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=${ARG_PARENT_TARGET}
                              -DCMAKE_INSTALL_DO_STRIP=1
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    set_target_properties(install-${ARG_PARENT_TARGET} PROPERTIES
                          FOLDER "Compiler-RT Misc")
    set_target_properties(install-${ARG_PARENT_TARGET}-stripped PROPERTIES
                          FOLDER "Compiler-RT Misc")
    add_dependencies(install-compiler-rt install-${ARG_PARENT_TARGET})
    add_dependencies(install-compiler-rt-stripped install-${ARG_PARENT_TARGET}-stripped)
  endif()

  # We only want to generate per-library install targets if you aren't using
  # an IDE because the extra targets get cluttered in IDEs.
  if(NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-${name}
                      DEPENDS ${name}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=${name}
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    add_custom_target(install-${name}-stripped
                      DEPENDS ${name}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=${name}
                              -DCMAKE_INSTALL_DO_STRIP=1
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    # If you have a parent target specified, we bind the new install target
    # to the parent install target.
    if(LIB_PARENT_TARGET)
      add_dependencies(install-${LIB_PARENT_TARGET} install-${name})
      add_dependencies(install-${LIB_PARENT_TARGET}-stripped install-${name}-stripped)
    endif()
  endif()
endfunction()
