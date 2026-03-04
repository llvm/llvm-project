##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

# Test execution is wrapped here because add_test only allows running
# one command at a time.

# FIXME: It would be better to use llvm-lit and parse RUN lines from
# individual tests.

execute_process(COMMAND
  ${CLANG_BIN} -O3 -S -cl-std=CL2.0
  -target amdgcn-amd-amdhsa -mcpu=${TEST_CPU}
  -Xclang -finclude-default-header
  --rocm-path=${BINARY_DIR}
  -mllvm -amdgpu-simplify-libcall=0
  ${COMPILE_FLAGS}
  -o ${OUTPUT_FILE} ${INPUT_FILE}
  RESULT_VARIABLE CLANG_RESULT
  ERROR_VARIABLE CLANG_ERR)
if(CLANG_RESULT)
  message(FATAL_ERROR "Error compiling test: ${CLANG_ERR}")
endif()

execute_process(COMMAND ${FILECHECK_BIN} -v --enable-var-scope
  --allow-unused-prefixes
  --dump-input=fail
  --dump-input-filter=all
  ${INPUT_FILE} --input-file ${OUTPUT_FILE}
  --check-prefixes=CHECK,${EXTRA_CHECK_PREFIX}
  RESULT_VARIABLE FILECHECK_RESULT
  ERROR_VARIABLE FILECHECK_ERROR)
if(FILECHECK_RESULT)
  message(FATAL_ERROR "Error in test output: ${FILECHECK_ERROR}")
endif()
