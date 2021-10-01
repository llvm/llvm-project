get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  FIROptimizer
  MLIRLLVMIR
  ${dialect_libs}
)

add_flang_unittest(FlangLoweringTests
  RTBuilder.cpp
)

target_link_libraries(FlangLoweringTests
  PRIVATE
  ${LIBS})
