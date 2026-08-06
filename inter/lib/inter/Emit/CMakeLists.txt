add_mlir_library(InterEmit
  IgaTextEmitter.cpp

  LINK_LIBS PUBLIC
  InterXeMachine
  MLIRFuncDialect
  MLIRIR
)
