##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

if(OCL_CONFORMANCE_HOME)
  file(GLOB files
#    ${OCL_CONFORMANCE_HOME}/integer_ops/*.cl
#    ${OCL_CONFORMANCE_HOME}/math_brute_force/*.cl
    ${OCL_CONFORMANCE_HOME}/commonfns/*.cl
    ${OCL_CONFORMANCE_HOME}/geometrics/*.cl
#    ${OCL_CONFORMANCE_HOME}/select/*.cl
  )
  foreach(file ${files})
    file(RELATIVE_PATH rfile ${OCL_CONFORMANCE_HOME} ${file})
    clang_opencl_test_file(${OCL_CONFORMANCE_HOME} ${rfile})
  endforeach()
endif()
