set(LLVM_TARGET_DEFINITIONS python_test_ops.td)
aiir_tablegen(lib/PythonTestDialect.h.inc -gen-dialect-decls)
aiir_tablegen(lib/PythonTestDialect.cpp.inc -gen-dialect-defs)
aiir_tablegen(lib/PythonTestOps.h.inc -gen-op-decls)
aiir_tablegen(lib/PythonTestOps.cpp.inc -gen-op-defs)
aiir_tablegen(lib/PythonTestAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(lib/PythonTestAttributes.cpp.inc -gen-attrdef-defs)
aiir_tablegen(lib/PythonTestTypes.h.inc -gen-typedef-decls)
aiir_tablegen(lib/PythonTestTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(AIIRPythonTestIncGen)

add_subdirectory(lib)

set(AIIR_PYTHON_TEST_DEPENDS AIIRPythonModules aiir-runner aiir_c_runner_utils aiir_runner_utils)
if(NOT AIIR_STANDALONE_BUILD)
  list(APPEND AIIR_PYTHON_TEST_DEPENDS FileCheck count not)
endif()
add_lit_testsuite(check-aiir-python "Running the AIIR Python regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${AIIR_PYTHON_TEST_DEPENDS}
)
