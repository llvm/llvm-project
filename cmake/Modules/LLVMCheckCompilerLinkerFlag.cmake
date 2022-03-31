include(CMakePushCheckState)

include(CheckCompilerFlag OPTIONAL)

if(NOT COMMAND check_compiler_flag)
  include(CheckCCompilerFlag)
  include(CheckCXXCompilerFlag)
endif()

function(llvm_check_compiler_linker_flag lang flag out_var)
  # If testing a flag with check_c_compiler_flag, it gets added to the compile
  # command only, but not to the linker command in that test. If the flag
  # is vital for linking to succeed, the test would fail even if it would
  # have succeeded if it was included on both commands.
  #
  # Therefore, try adding the flag to CMAKE_REQUIRED_FLAGS, which gets
  # added to both compiling and linking commands in the tests.

  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${flag}")
  if(COMMAND check_compiler_flag)
    check_compiler_flag("${lang}" "" ${out_var})
  else()
    # Until the minimum CMAKE version is 3.19
    # cmake builtin compatible, except we assume lang is C or CXX
    if("${lang}" STREQUAL "C")
      check_c_compiler_flag("" ${out_var})
    elseif("${lang}" STREQUAL "CXX")
      check_cxx_compiler_flag("" ${out_var})
    else()
      message(FATAL_ERROR "\"${lang}\" is not C or CXX")
    endif()
  endif()
  cmake_pop_check_state()
endfunction()
