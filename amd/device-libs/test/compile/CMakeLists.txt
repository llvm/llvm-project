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

function(add_compile_test test_name func_name script test_cpu extra_check_prefixes)
  add_test(NAME ${test_name}__${test_cpu}
    COMMAND ${CMAKE_COMMAND}
     -DCLANG_BIN=$<TARGET_FILE:clang>
     -DBINARY_DIR=${PROJECT_BINARY_DIR}
     -DFILECHECK_BIN=${FILECHECK_BIN}
     -DOUTPUT_FILE=output.${name}.${test_cpu}.s
     -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${func_name}.cl
     -DTEST_CPU=${test_cpu}
     -DEXTRA_CHECK_PREFIX=${extra_check_prefixes}
     -P ${script})
endfunction()


# Add constant folding tests
function(add_constant_fold_test name test_cpu)
  add_compile_test(constant_fold_${name} ${name}
                  ${CMAKE_CURRENT_SOURCE_DIR}/RunConstantFoldTest.cmake ${test_cpu} CHECK)
endfunction()

# Add full to ISA compile tests
function(add_isa_test name test_cpu)
  string(TOUPPER ${test_cpu} check_prefix)
  add_compile_test(compile_${name} ${name}
                   ${CMAKE_CURRENT_SOURCE_DIR}/RunCompileTest.cmake ${test_cpu} "${check_prefix},GCN")
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


foreach(gpu gfx600 gfx700 gfx803)
  add_isa_test(fract ${gpu})
  add_isa_test(native_rcp ${gpu})
  add_isa_test(native_rsqrt ${gpu})
  add_isa_test(native_log ${gpu})
  add_isa_test(native_exp ${gpu})
endforeach()
