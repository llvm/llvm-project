set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-unused-parameter")

add_flang_library(FortranLower
  ComplexExpr.cpp
  ConvertType.cpp
  DoLoopHelper.cpp
  OpenMP.cpp
  PFTBuilder.cpp

  DEPENDS
  FIROpsIncGen
  MLIROpAsmInterfaceIncGen
  MLIRControlFlowInterfaces
  MLIRLinalgStructuredOpsInterfaceIncGen

  LINK_COMPONENTS
  Support
)
