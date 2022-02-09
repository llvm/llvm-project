get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FortranLower
  Bridge.cpp
  CallInterface.cpp
  Coarray.cpp
  ConvertExpr.cpp
  ConvertType.cpp
  ConvertVariable.cpp
  Mangler.cpp
  OpenACC.cpp
  OpenMP.cpp
  PFTBuilder.cpp
  Runtime.cpp
  SymbolMap.cpp

  DEPENDS
  FIRDialect
  FIRBuilder
  FIRSupport
  FIRTransforms
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRBuilder
  FIRSupport
  FIRTransforms
  ${dialect_libs}
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRSCFToControlFlow
  MLIRStandard

  LINK_COMPONENTS
  Support
)
