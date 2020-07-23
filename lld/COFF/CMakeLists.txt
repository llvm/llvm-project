set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(COFFOptionsTableGen)

add_lld_library(lldCOFF
  Chunks.cpp
  DebugTypes.cpp
  DLL.cpp
  Driver.cpp
  DriverUtils.cpp
  ICF.cpp
  InputFiles.cpp
  LLDMapFile.cpp
  LTO.cpp
  MapFile.cpp
  MarkLive.cpp
  MinGW.cpp
  PDB.cpp
  SymbolTable.cpp
  Symbols.cpp
  Writer.cpp

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BinaryFormat
  Core
  DebugInfoCodeView
  DebugInfoDWARF
  DebugInfoMSF
  DebugInfoPDB
  Demangle
  LibDriver
  LTO
  MC
  Object
  Option
  Passes
  Support
  WindowsManifest

  LINK_LIBS
  lldCommon
  ${LLVM_PTHREAD_LIB}

  DEPENDS
  COFFOptionsTableGen
  intrinsics_gen
  )
