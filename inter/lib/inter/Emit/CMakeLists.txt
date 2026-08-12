include_directories(SYSTEM ${INTER_GED_INCLUDE_DIRS})

add_mlir_library(InterEmit
  GedBinaryEmitter.cpp
  XeMachineLowering.cpp
  ZebinEmitter.cpp

  LINK_LIBS PUBLIC
  InterXeMachine
  MLIRFuncDialect
  MLIRIR
)

target_sources(InterEmit PRIVATE $<TARGET_OBJECTS:GEDLibrary>)
