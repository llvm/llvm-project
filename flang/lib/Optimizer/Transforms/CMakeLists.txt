add_flang_library(FIRTransforms
  AbstractResult.cpp
  AffinePromotion.cpp
  AffineDemotion.cpp
  CharacterConversion.cpp
  ArrayValueCopy.cpp
  ExternalNameConversion.cpp
  MemoryAllocation.cpp
  MemRefDataFlowOpt.cpp
  RewriteLoop.cpp

  DEPENDS
  FIRBuilder
  FIRDialect
  FIRSupport
  FIROptTransformsPassIncGen

  LINK_LIBS
  FIRBuilder
  FIRDialect
  MLIRAffineUtils
  MLIRLLVMIR
  MLIROpenACC
  MLIROpenMP
  FIRSupport
)
