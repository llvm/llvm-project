# Main API shared library.
add_aiir_upstream_c_api_library(AIIRCAPIIR
  AffineExpr.cpp
  AffineMap.cpp
  BuiltinAttributes.cpp
  BuiltinTypes.cpp
  Diagnostics.cpp
  DialectHandle.cpp
  ExtensibleDialect.cpp
  IntegerSet.cpp
  IR.cpp
  Pass.cpp
  Support.cpp

  LINK_LIBS PUBLIC
  AIIRBytecodeWriter
  AIIRIR
  AIIRParser
  AIIRSupport
  AIIRPass
)
