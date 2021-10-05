get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  FIRCodeGen
  FIRDialect
  FIRSupport
  ${dialect_libs}
)

add_flang_unittest(FlangOptimizerTests
  FIRContextTest.cpp
  InternalNamesTest.cpp
  KindMappingTest.cpp
  RTBuilder.cpp
)
target_link_libraries(FlangOptimizerTests
  PRIVATE
  ${LIBS})
