add_aiir_unittest(AIIRPresburgerTests
  BarvinokTest.cpp
  FractionTest.cpp
  GeneratingFunctionTest.cpp
  IntegerPolyhedronTest.cpp
  IntegerRelationTest.cpp
  LinearTransformTest.cpp
  MatrixTest.cpp
  Parser.h
  ParserTest.cpp
  PresburgerSetTest.cpp
  PresburgerRelationTest.cpp
  PresburgerSpaceTest.cpp
  PWMAFunctionTest.cpp
  QuasiPolynomialTest.cpp
  SimplexTest.cpp
  UtilsTest.cpp
)

aiir_target_link_libraries(AIIRPresburgerTests
  PRIVATE AIIRPresburger
  AIIRAffineAnalysis
  AIIRParser
  )
