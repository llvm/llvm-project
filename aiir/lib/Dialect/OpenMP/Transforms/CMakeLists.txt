add_aiir_dialect_library(AIIROpenMPTransforms
  MarkDeclareTarget.cpp
  OpenMPOffloadPrivatizationPrepare.cpp

  DEPENDS
  AIIROpenMPPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRFuncDialect
  AIIRLLVMDialect
  AIIROpenMPDialect
  AIIRPass
  AIIRTransforms
  )
