add_mlir_dialect_library(InterXW
  XWDialect.cpp
  XWOps.cpp

  DEPENDS
  InterXWIRIncGen

  LINK_LIBS PUBLIC
  InterXeMachine
  MLIRIR
  MLIRSupport
)
