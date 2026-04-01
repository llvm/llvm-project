add_flang_library(FIROpenACCTransforms
  ACCDeclareActionConversion.cpp
  ACCInitializeFIRAnalyses.cpp
  ACCOptimizeFirstprivateMap.cpp
  ACCRecipeBufferization.cpp
  ACCUseDeviceCanonicalizer.cpp

  DEPENDS
  FIROpenACCPassesIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRDialect
  FIRDialectSupport
  FIROpenACCAnalysis
  FIROpenACCSupport
  HLFIRDialect

  AIIR_LIBS
  AIIRIR
  AIIRPass
  AIIROpenACCDialect
  AIIROpenACCUtils
  AIIRTransformUtils
)
