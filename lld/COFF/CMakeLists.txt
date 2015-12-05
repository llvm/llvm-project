set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(COFFOptionsTableGen)

add_llvm_library(lldCOFF
  Chunks.cpp
  DLL.cpp
  Driver.cpp
  DriverUtils.cpp
  Error.cpp
  ICF.cpp
  InputFiles.cpp
  MarkLive.cpp
  ModuleDef.cpp
  PDB.cpp
  SymbolTable.cpp
  Symbols.cpp
  Writer.cpp

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LTO
  LibDriver
  Object
  MC
  MCDisassembler
  Target
  Option
  Support
  )

add_dependencies(lldCOFF COFFOptionsTableGen)
