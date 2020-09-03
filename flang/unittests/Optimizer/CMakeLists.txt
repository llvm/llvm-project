get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  FIROptimizer
  ${dialect_libs}
)

add_flang_unittest(FlangOptimizerTests
  InternalNamesTest.cpp
)
target_link_libraries(FlangOptimizerTests
  PRIVATE
  ${LIBS})
