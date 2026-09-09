add_clang_library(MLIRCIRTargetLowering
  CIRABIRewriteContext.cpp
  CIRCXXABI.cpp
  LowerModule.cpp
  LowerItaniumCXXABI.cpp
  TargetLoweringInfo.cpp
  Targets/AMDGPU.cpp
  Targets/NVPTX.cpp
  Targets/SPIRV.cpp

  DEPENDS
  clangBasic

  DISABLE_PCH_REUSE # PCH contains private headers
  PRECOMPILE_HEADERS
  [["pch.h"]]

  LINK_COMPONENTS
  TargetParser

  LINK_LIBS PUBLIC

  clangBasic
  MLIRABI
  MLIRIR
  MLIRPass
  MLIRDLTIDialect
  MLIRCIR
  MLIRCIRInterfaces
  MLIRLLVMDialect
)
