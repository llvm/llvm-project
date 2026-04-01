set(LLVM_TARGET_DEFINITIONS enums.td)
aiir_tablegen(EnumsGenTest.h.inc -gen-enum-decls)
aiir_tablegen(EnumsGenTest.cpp.inc -gen-enum-defs)
add_public_tablegen_target(AIIRTableGenEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS passes.td)
aiir_tablegen(PassGenTest.h.inc -gen-pass-decls -name TableGenTest)
add_public_tablegen_target(AIIRTableGenTestPassIncGen)

add_aiir_unittest(AIIRTableGenTests
  EnumsGenTest.cpp
  FormatTest.cpp
  OpBuildGen.cpp
  PassGenTest.cpp
)

add_dependencies(AIIRTableGenTests AIIRTableGenEnumsIncGen)
add_dependencies(AIIRTableGenTests AIIRTableGenTestPassIncGen)
add_dependencies(AIIRTableGenTests AIIRTestDialect)

target_include_directories(AIIRTableGenTests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../test/lib/Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../test/lib/Dialect/Test
)

target_link_libraries(AIIRTableGenTests
  PRIVATE LLVMTableGen AIIRTableGen AIIRIR
  PUBLIC AIIRTestDialect
)
