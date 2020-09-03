get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  MLIRLLVMIR
  ${dialect_libs}
)

add_flang_unittest(FlangLoweringOpenMPTests
	OpenMPLoweringTest.cpp
)
target_link_libraries(FlangLoweringOpenMPTests
  PRIVATE
  ${LIBS})
