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

macro(append_have_file_definition filename varname list)
  check_include_file("${filename}" "${varname}")
  if (NOT ${varname})
    set("${varname}" 0)
  endif()
  list(APPEND ${list} "${varname}=${${varname}}")
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
      try_compile_only(CAN_TARGET_${arch} ${TARGET_${arch}_CFLAGS})
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
  check_symbol_exists(__s390x__ "" __S390X)
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
  elseif(__S390X)
    add_default_target_arch(s390x)
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
      message(FATAL_ERROR "llvm-config not found: specify LLVM_CONFIG_PATH")
    endif()
  endif()
  execute_process(
    COMMAND ${LLVM_CONFIG_PATH} "--obj-root" "--bindir" "--libdir" "--src-root"
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

  set(LLVM_BINARY_DIR ${BINARY_DIR} CACHE PATH "Path to LLVM build tree")
  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
  set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")

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
  list(GET TARGET_TRIPLE_LIST 1 COMPILER_RT_DEFAULT_TARGET_OS)
  list(LENGTH TARGET_TRIPLE_LIST TARGET_TRIPLE_LIST_LENGTH)
  if(TARGET_TRIPLE_LIST_LENGTH GREATER 2)
    list(GET TARGET_TRIPLE_LIST 2 COMPILER_RT_DEFAULT_TARGET_ABI)
  endif()
  # Determine if test target triple is specified explicitly, and doesn't match the
  # default.
  if(NOT COMPILER_RT_DEFAULT_TARGET_TRIPLE STREQUAL TARGET_TRIPLE)
    set(COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE TRUE)
  else()
    set(COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE FALSE)
  endif()
endmacro()
