get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  FIRBuilder
  FIRCodeGen
  FIRDialect
  FIRSupport
  ${dialect_libs}
)

add_flang_unittest(FlangOptimizerTests
  Builder/CharacterTest.cpp
  Builder/ComplexTest.cpp
  Builder/DoLoopHelperTest.cpp
  Builder/FIRBuilderTest.cpp
  Builder/Runtime/AssignTest.cpp
  Builder/Runtime/CharacterTest.cpp
  Builder/Runtime/DerivedTest.cpp
  Builder/Runtime/NumericTest.cpp
  Builder/Runtime/RaggedTest.cpp
  Builder/Runtime/ReductionTest.cpp
  Builder/Runtime/TransformationalTest.cpp
  FIRContextTest.cpp
  InternalNamesTest.cpp
  KindMappingTest.cpp
  RTBuilder.cpp
)
target_link_libraries(FlangOptimizerTests
  PRIVATE
  ${LIBS})
