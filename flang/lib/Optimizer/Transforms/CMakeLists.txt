get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRTransforms
  ControlFlowConverter.cpp
  CSE.cpp
  Inliner.cpp
  MemToReg.cpp
  RaiseToAffine.cpp
  RewriteLoop.cpp

  DEPENDS
  FIROpsIncGen
  ${dialect_libs}

  LINK_LIBS
  FIRAnalysis
  FIRDialect
  FIRSupport
)
