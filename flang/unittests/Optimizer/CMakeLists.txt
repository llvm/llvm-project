get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  FIRBuilder
  FIRCodeGen
  FIRDialect
  FIRDialectSupport
  FIRSupport
  HLFIRDialect
  ${dialect_libs}
  ${extension_libs}
  LLVMTargetParser
)

add_flang_unittest(FlangOptimizerTests
  Builder/CharacterTest.cpp
  Builder/ComplexTest.cpp
  Builder/DoLoopHelperTest.cpp
  Builder/FIRBuilderTest.cpp
  Builder/HLFIRToolsTest.cpp
  FIRContextTest.cpp
  FIRTypesTest.cpp
  FortranVariableTest.cpp
  InternalNamesTest.cpp
  KindMappingTest.cpp
  RTBuilder.cpp
DEPENDS
  FIRDialect
  FIRSupport
  HLFIRDialect
  ${dialect_libs})

target_link_libraries(FlangOptimizerTests
  PRIVATE
  ${LIBS})
