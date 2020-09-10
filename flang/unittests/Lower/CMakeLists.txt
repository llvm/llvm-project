get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  FIROptimizer
  MLIRLLVMIR
  ${dialect_libs}
)

add_flang_unittest(FlangLoweringOpenMPTests
  IntervalSet.cpp
  OpenMPLoweringTest.cpp
  RTBuilder.cpp  
)
target_link_libraries(FlangLoweringOpenMPTests
  PRIVATE
  ${LIBS})
