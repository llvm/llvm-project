set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(ELFOptionsTableGen)

add_lld_library(lldELF
  Driver.cpp
  DriverUtils.cpp
  EhFrame.cpp
  Error.cpp
  GdbIndex.cpp
  ICF.cpp
  InputFiles.cpp
  InputSection.cpp
  LTO.cpp
  LinkerScript.cpp
  MarkLive.cpp
  Memory.cpp
  Mips.cpp
  OutputSections.cpp
  Relocations.cpp
  ScriptParser.cpp
  Strings.cpp
  SymbolListFile.cpp
  SymbolTable.cpp
  Symbols.cpp
  SyntheticSections.cpp
  Target.cpp
  Thunks.cpp
  Writer.cpp

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  BitReader
  BitWriter
  Codegen
  Core
  DebugInfoDWARF
  Demangle
  IPO
  Linker
  LTO
  Object
  Option
  Passes
  MC
  Support
  Target
  TransformUtils

  LINK_LIBS
  lldConfig
  lldCore
  ${PTHREAD_LIB}

  DEPENDS
  ELFOptionsTableGen
  intrinsics_gen
  )
