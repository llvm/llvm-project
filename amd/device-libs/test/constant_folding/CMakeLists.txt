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

function(add_constant_fold_test name)
  add_test(NAME constant_fold_${name}
    COMMAND ${CMAKE_COMMAND}
     -DCLANG_BIN=$<TARGET_FILE:clang>
     -DBINARY_DIR=${PROJECT_BINARY_DIR}
     -DFILECHECK_BIN=${FILECHECK_BIN}
     -DOUTPUT_FILE=output.${name}.ll
     -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}.cl
     -DTEST_CPU=gfx900
     -P ${CMAKE_CURRENT_SOURCE_DIR}/RunConstantFoldTest.cmake)
endfunction()

add_constant_fold_test(lgamma_r)
