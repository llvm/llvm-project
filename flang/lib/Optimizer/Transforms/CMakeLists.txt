add_flang_library(FIRTransforms
  AffinePromotion.cpp
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
