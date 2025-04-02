#===-- unittests/CMakeLists.txt --------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# LLVM uses a modified version of GTest that uses LLVMSupport for console
# output. Therefore it also needs to include files from LLVM. Unfortunately,
# LLVM/GTest doesn't add the include search path itself. Limiting the scope
# using target_include_directories does not work because with
# LLVM_INSTALL_GTEST=ON, as llvm_gtest is an IMPORT library.
include_directories("${LLVM_INCLUDE_DIR}" "${LLVM_MAIN_INCLUDE_DIR}")

# Add GTest if not already present.
# Using a function so LLVM_SUBPROJECT_TITLE does not propagate.
function (build_gtest)
  set(LLVM_SUBPROJECT_TITLE "Third-Party/Google Test")
  add_subdirectory("${LLVM_THIRD_PARTY_DIR}/unittest" "${CMAKE_CURRENT_BINARY_DIR}/third-party/unittest")
endfunction ()
if (NOT TARGET llvm_gtest)
  build_gtest()
endif ()

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()


# Target that depends on all unittests
add_custom_target(FlangRTUnitTests)
set_target_properties(FlangRTUnitTests PROPERTIES FOLDER "Flang-RT/Meta")


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


function(add_flangrt_unittest test_dirname)
  cmake_parse_arguments(ARG
    ""
    ""
    "LINK_LIBS"
    ${ARGN})

  add_unittest(FlangRTUnitTests ${test_dirname} ${ARG_UNPARSED_ARGUMENTS})

  target_link_libraries(${test_dirname} PRIVATE ${ARG_LINK_LIBS})
  add_flangrt_unittest_offload_properties(${test_dirname})

  # Required because LLVMSupport is compiled with this option.
  # FIXME: According to CMake documentation, this is the default. Why is it
  #        needed? LLVM's add_unittest doesn't set it either.
  set_target_properties(${test_dirname}
      PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )
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

  if(NOT ARG_SLOW_TEST)
    add_dependencies(FlangRTUnitTests ${test_name}${suffix})
  endif()

  add_flangrt_unittest_offload_properties(${test_name}${suffix})
endfunction()

add_subdirectory(Evaluate)
add_subdirectory(Runtime)
