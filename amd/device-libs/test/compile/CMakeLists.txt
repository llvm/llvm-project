##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

if(TARGET FileCheck)
  set(FILECHECK_BIN $<TARGET_FILE:FileCheck>)
else()
  # FIXME: Is there a better way to get the binary directory?
  # FileCheck is also not normally installed, so it only really works
  # well with build directories by default.
  find_program(FILECHECK_BIN FileCheck
    HINTS ${LLVM_DIR}/../../../bin)
endif()

if(NOT FILECHECK_BIN)
  message(STATUS "FileCheck not found, not adding constant fold tests")
  return()
endif()

message(STATUS "Running constant fold tests")

function(add_compile_test test_name func_name)
  set(parse_options)
  set(one_value_args TEST_CPU FILE_NAME SCRIPT EXTRA_CHECK_PREFIXES)
  set(multi_value_args COMPILE_FLAGS)

  cmake_parse_arguments(COMPILE_TEST "${parse_options}" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  set(test_cpu ${COMPILE_TEST_TEST_CPU})
  set(file_name ${COMPILE_TEST_FILE_NAME})

  add_test(NAME ${test_name}__${test_cpu}
    COMMAND ${CMAKE_COMMAND}
     -DCLANG_BIN=$<TARGET_FILE:clang>
     -DBINARY_DIR=${PROJECT_BINARY_DIR}
     -DFILECHECK_BIN=${FILECHECK_BIN}
     -DOUTPUT_FILE=output.${test_name}.${test_cpu}.s
     -DINPUT_FILE=${file_name}
     -DTEST_CPU=${test_cpu}
     -DCOMPILE_FLAGS=${COMPILE_TEST_COMPILE_FLAGS}
     -DEXTRA_CHECK_PREFIX=${COMPILE_TEST_EXTRA_CHECK_PREFIXES}
     -P ${COMPILE_TEST_SCRIPT})
endfunction()


# Add constant folding tests
function(add_constant_fold_test name test_cpu)
  add_compile_test(constant_fold_${name} ${name}
    FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cl
    SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/RunConstantFoldTest.cmake
    TEST_CPU ${test_cpu}
    EXTRA_CHECK_PREFIXES CHECK
    ${ARGN})
endfunction()

# Add full to ISA compile tests
function(add_isa_test name test_cpu)
  string(TOUPPER ${test_cpu} check_prefix)
  add_compile_test(compile_${name} ${name}
    FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cl
    SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/RunCompileTest.cmake
    TEST_CPU ${test_cpu}
    EXTRA_CHECK_PREFIXES "${check_prefix},GCN"
    ${ARGN})
endfunction()


foreach(gpu gfx900 gfx1030)
  add_constant_fold_test(lgamma_r ${gpu})
endforeach()

foreach(gpu gfx803)
  add_isa_test(asin ${gpu})
  add_isa_test(atan2 ${gpu})
  add_isa_test(atan2pi ${gpu})
endforeach()

foreach(gpu gfx600)
  add_isa_test(frexp ${gpu})
endforeach()

foreach(gpu gfx900)
  # Test with default denormal enabled target
  add_isa_test(rsqrt ${gpu}
    EXTRA_CHECK_PREFIXES IEEE)
  add_isa_test(rsqrt_daz ${gpu}
    FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/rsqrt.cl
    COMPILE_FLAGS -cl-denorms-are-zero
    EXTRA_CHECK_PREFIXES DAZ)
endforeach()


foreach(gpu gfx600 gfx700 gfx803)
  add_isa_test(fract ${gpu})
  add_isa_test(native_rcp ${gpu})
  add_isa_test(native_rsqrt ${gpu})
  add_isa_test(native_log ${gpu})
  add_isa_test(native_exp ${gpu})
endforeach()
