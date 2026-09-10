include_directories(SYSTEM ${INTER_GED_INCLUDE_DIRS})

add_mlir_library(InterEmit
  GedBinaryEmitter.cpp
  IgaAssemblyEmitter.cpp
  XeMachineLowering.cpp
  ZebinEmitter.cpp

  LINK_LIBS PUBLIC
  IGA_SLIB
  InterXeMachine
  MLIRFuncDialect
  MLIRIR
)

target_sources(InterEmit PRIVATE $<TARGET_OBJECTS:GEDLibrary>)
