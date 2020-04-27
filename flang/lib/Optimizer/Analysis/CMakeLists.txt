get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRAnalysis
  IteratedDominanceFrontier.cpp

  DEPENDS
  FIROpsIncGen
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRSupport
)
