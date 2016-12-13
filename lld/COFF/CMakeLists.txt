set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(COFFOptionsTableGen)

if(NOT LLD_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lld_library(lldCOFF
  Chunks.cpp
  DLL.cpp
  Driver.cpp
  DriverUtils.cpp
  Error.cpp
  ICF.cpp
  InputFiles.cpp
  Librarian.cpp
  MarkLive.cpp
  ModuleDef.cpp
  PDB.cpp
  Strings.cpp
  SymbolTable.cpp
  Symbols.cpp
  Writer.cpp

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  LTO
  LibDriver
  Object
  MC
  MCDisassembler
  Target
  Option
  Support

  LINK_LIBS
  lldCore
  lldSupport
  ${PTHREAD_LIB}

  DEPENDS
  COFFOptionsTableGen
  ${tablegen_deps}
  )
