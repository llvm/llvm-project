#===-- unittests/CMakeLists.txt --------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Target that depends on all unittests
add_custom_target(FlangRTUnitTests)
set_target_properties(FlangRTUnitTests PROPERTIES FOLDER "Flang-RT/Meta")

# LLVM uses a modified version of GTest that uses LLVMSupport for console
# output. We are using the pre-compiled GTest library from the LLVM build,
# if available. Otherwise, do nothing.

if (CMAKE_CROSSCOMPILING)
  # TODO: It is possible that LLVM_GTEST_RUN_UNDER defines an emulator or
  #       ssh remote command invocation; for this case provide an option to
  #       enable unittests.
  message(STATUS "Flang-RT unittests disabled because we are cross-compiling")
  return ()
endif ()

if (NOT TARGET llvm_gtest)
  message(WARNING "Flang-RT unittests disabled due to GTest being unavailable; "
                  "Try LLVM_INSTALL_GTEST=ON for the LLVM build")
  return ()
endif ()


add_dependencies(flang-rt-test-depends
  FlangRTUnitTests
  flang_rt.runtime.unittest
)

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()


function(add_flangrt_unittest_offload_properties target)
  # Set CUDA_RESOLVE_DEVICE_SYMBOLS.
  if (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "CUDA")
    set_target_properties(${target}
      PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
  endif()
  # Enable OpenMP offload during linking. We may need to replace
  # LINK_OPTIONS with COMPILE_OPTIONS when there are OpenMP offload
  # unittests.
  #
  # FIXME: replace 'native' in --offload-arch option with the list
  #        of targets that Fortran Runtime was built for.
  if (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "OpenMP")
    set_target_properties(${target}
      PROPERTIES LINK_OPTIONS
      "-fopenmp;--offload-arch=native"
      )
  endif()
endfunction()

# flang-rt on Windows requires compiler-rt for some symbols. For binaries built
# with flang this dependency is added by the flang driver, but since the unit
# tests are built with clang we need to add the dependency manually.
function(add_flangrt_dependent_libs target)
  if (MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (FLANG_RT_BUILTINS_LIBRARY)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:-Xclang>" "$<$<COMPILE_LANGUAGE:CXX,C>:--dependent-lib=${FLANG_RT_BUILTINS_LIBRARY}>")
    endif ()
  endif ()
  if (MSVC AND CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
    if (FLANG_RT_BUILTINS_LIBRARY)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:Fortran>:-Xflang>" "$<$<COMPILE_LANGUAGE:Fortran>:--dependent-lib=${FLANG_RT_BUILTINS_LIBRARY}>")
    else ()
      message(WARNING "Did not find libclang_rt.builtins.lib.
        LLVM may emit builtins that are not implemented in msvcrt/ucrt and
        instead falls back to builtins from Compiler-RT. Linking with ${tgtname}
        may result in a linker error.")
    endif ()
  endif ()
endfunction()


function(add_flangrt_unittest test_dirname)
  cmake_parse_arguments(ARG
    ""
    ""
    "LINK_LIBS"
    ${ARGN})

  add_unittest(FlangRTUnitTests ${test_dirname} ${ARG_UNPARSED_ARGUMENTS})

  target_link_libraries(${test_dirname} PRIVATE ${ARG_LINK_LIBS})
  add_flangrt_unittest_offload_properties(${test_dirname})
  add_flangrt_dependent_libs(${test_dirname})
endfunction()

function(add_flangrt_nongtest_unittest test_name)
  cmake_parse_arguments(ARG
    "SLOW_TEST"
    ""
    "LINK_LIBS"
    ${ARGN})

  if(ARG_SLOW_TEST)
      set(suffix .slow)
  else()
      set(suffix .test)
  endif()

  add_executable(${test_name}${suffix} EXCLUDE_FROM_ALL ${ARG_UNPARSED_ARGUMENTS})
  set_target_properties(${test_name}${suffix} PROPERTIES FOLDER "Flang-RT/Tests/Unit")

  target_link_libraries(${test_name}${suffix} PRIVATE NonGTestTesting ${ARG_LINK_LIBS})
  add_flangrt_dependent_libs(${test_name}${suffix})

  if(NOT ARG_SLOW_TEST)
    add_dependencies(FlangRTUnitTests ${test_name}${suffix})
  endif()

  add_flangrt_unittest_offload_properties(${test_name}${suffix})
endfunction()

add_subdirectory(Evaluate)
add_subdirectory(Runtime)
