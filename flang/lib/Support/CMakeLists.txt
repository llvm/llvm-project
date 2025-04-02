add_flang_library(FortranSupport
  Timing.cpp

  LINK_COMPONENTS
  Support

  MLIR_LIBS
  MLIRSupport
)
