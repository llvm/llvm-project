add_flang_library(FIRTransforms
  Inliner.cpp
  ExternalNameConversion.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  FIROptTransformsPassIncGen

  LINK_LIBS
  FIRDialect
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIROpenACC
  MLIROpenMP
  FIRSupport
)
