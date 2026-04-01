add_aiir_dialect_library(AIIRLLVMIRTransforms
  AddComdats.cpp
  DIExpressionLegalization.cpp
  DIExpressionRewriter.cpp
  DIScopeForLLVMFuncOp.cpp
  InlinerInterfaceImpl.cpp
  LegalizeForExport.cpp
  OptimizeForNVVM.cpp
  RequestCWrappers.cpp
  UseDefaultVisibilityPass.cpp

  DEPENDS
  AIIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRFuncDialect
  AIIRLLVMDialect
  AIIRPass
  AIIRTransforms
  AIIRNVVMDialect
  AIIROpenMPDialect
  )
