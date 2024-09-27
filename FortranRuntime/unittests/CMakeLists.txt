#===-- unittests/CMakeLists.txt --------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

function (build_gtest)
  include_directories("${LLVM_INCLUDE_DIR}" "${LLVM_MAIN_INCLUDE_DIR}")
  set(LLVM_SUBPROJECT_TITLE "Third-Party/Google Test")
  add_subdirectory(${LLVM_THIRD_PARTY_DIR}/unittest ${CMAKE_CURRENT_BINARY_DIR}/third-party/unittest)
endfunction ()

# Add GTest if not already present
if (NOT TARGET llvm_gtest)
  build_gtest()
endif ()

# Target that depends on all unittests
add_custom_target(FortranRuntimeUnitTests)
set_target_properties(FortranRuntimeUnitTests PROPERTIES FOLDER "Fortran Runtime/Tests")

function(add_fortranruntime_unittest test_dirname)
  cmake_parse_arguments(ARG
    ""
    ""
    "LINK_LIBS"
    ${ARGN})

  add_unittest(FortranRuntimeUnitTests ${test_dirname} ${ARG_UNPARSED_ARGUMENTS})
  target_include_directories(${test_dirname} PRIVATE "${FORTRANRUNTIME_SOURCE_DIR}/include")
  target_link_libraries(${test_dirname} PRIVATE ${ARG_LINK_LIBS})
endfunction()

function(add_fortranruntime_nongtest_unittest test_name)
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

  add_executable(${test_name}${suffix} ${ARG_UNPARSED_ARGUMENTS})
  target_link_libraries(${test_name}${suffix} PRIVATE NonGTestTesting ${ARG_LINK_LIBS})
  target_include_directories(${test_name}${suffix} PRIVATE "${FORTRANRUNTIME_SOURCE_DIR}/include")
  set_target_properties(${test_name}${suffix} PROPERTIES FOLDER "Fortran Runtime/Tests/Unit")

  if(NOT ARG_SLOW_TEST)
    add_dependencies(FortranRuntimeUnitTests ${test_name}${suffix})
  endif()
endfunction()

add_subdirectory(Common)
add_subdirectory(Decimal)
add_subdirectory(Evaluate)
add_subdirectory(Runtime)
