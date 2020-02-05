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
  check_symbol_exists(__powerpc64__ "" __PPC64)
  check_symbol_exists(__powerpc64le__ "" __PPC64LE)
  check_symbol_exists(__riscv "" __RISCV)
  check_symbol_exists(__s390x__ "" __S390X)
  check_symbol_exists(__sparc "" __SPARC)
  check_symbol_exists(__sparcv9 "" __SPARCV9)
  check_symbol_exists(__wasm32__ "" __WEBASSEMBLY32)
  check_symbol_exists(__wasm64__ "" __WEBASSEMBLY64)
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
  elseif(__PPC64)
    add_default_target_arch(powerpc64)
  elseif(__PPC64LE)
    add_default_target_arch(powerpc64le)
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
  endif()
endmacro()

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
    set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
    set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
    set(LLVM_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Paths to LLVM headers")

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
    else()
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

    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    # Get some LLVM variables from LLVMConfig.
    include("${LLVM_CMAKE_PATH}/LLVMConfig.cmake")

    set(LLVM_LIBRARY_OUTPUT_INTDIR
      ${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
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
  # Determine if test target triple is specified explicitly, and doesn't match the
  # default.
  if(NOT COMPILER_RT_DEFAULT_TARGET_TRIPLE STREQUAL TARGET_TRIPLE)
    set(COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE TRUE)
  else()
    set(COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE FALSE)
  endif()
endmacro()

# Filter out generic versions of routines that are re-implemented in
# architecture specific manner.  This prevents multiple definitions of the
# same symbols, making the symbol selection non-deterministic.
function(filter_builtin_sources output_var exclude_or_include excluded_list)
  if(exclude_or_include STREQUAL "EXCLUDE")
    set(filter_action GREATER)
    set(filter_value -1)
  elseif(exclude_or_include STREQUAL "INCLUDE")
    set(filter_action LESS)
    set(filter_value 0)
  else()
    message(FATAL_ERROR "filter_builtin_sources called without EXCLUDE|INCLUDE")
  endif()

  set(intermediate ${ARGN})
  foreach (_file ${intermediate})
    get_filename_component(_name_we ${_file} NAME_WE)
    list(FIND ${excluded_list} ${_name_we} _found)
    if(_found ${filter_action} ${filter_value})
      list(REMOVE_ITEM intermediate ${_file})
    elseif(${_file} MATCHES ".*/.*\\.S" OR ${_file} MATCHES ".*/.*\\.c")
      get_filename_component(_name ${_file} NAME)
      string(REPLACE ".S" ".c" _cname "${_name}")
      list(REMOVE_ITEM intermediate ${_cname})
    endif ()
  endforeach ()
  set(${output_var} ${intermediate} PARENT_SCOPE)
endfunction()

function(get_compiler_rt_target arch variable)
  string(FIND ${COMPILER_RT_DEFAULT_TARGET_TRIPLE} "-" dash_index)
  string(SUBSTRING ${COMPILER_RT_DEFAULT_TARGET_TRIPLE} ${dash_index} -1 triple_suffix)
  if(COMPILER_RT_DEFAULT_TARGET_ONLY)
    # Use exact spelling when building only for the target specified to CMake.
    set(target "${COMPILER_RT_DEFAULT_TARGET_TRIPLE}")
  elseif(ANDROID AND ${arch} STREQUAL "i386")
    set(target "i686${COMPILER_RT_OS_SUFFIX}${triple_suffix}")
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
