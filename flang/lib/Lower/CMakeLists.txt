get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FortranLower
  CharacterExpr.cpp
  CharacterRuntime.cpp
  Coarray.cpp
  ComplexExpr.cpp
  ConvertType.cpp
  ConvertExpr.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp
  IntrinsicCall.cpp
  IO.cpp
  Mangler.cpp
  OpenACC.cpp
  OpenMP.cpp
  PFTBuilder.cpp

  DEPENDS
  FIROptimizer
  ${dialect_libs}

  LINK_LIBS
  FIROptimizer
  ${dialect_libs}
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRStandard

  LINK_COMPONENTS
  Support
)
