add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  CIRDataLayout.cpp
  CIROpenCLAttrs.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRTypeConstraintsIncGen
  MLIRCIREnumsGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen

  # CIRDialect.h is a public header and the most commonly-included header
  # across the rest of clang/lib/CIR; building it here (rather than as a
  # private per-library PCH) lets every other CIR library that links
  # MLIRCIR automatically reuse this PCH too.
  PRECOMPILE_HEADERS
  [["clang/CIR/Dialect/IR/CIRDialect.h"]]

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIRInterfaces
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRCIRInterfaces
  clangAST
  )
