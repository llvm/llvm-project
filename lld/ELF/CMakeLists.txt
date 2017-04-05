set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(ELFOptionsTableGen)

if(NOT LLD_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lld_library(lldELF
  Driver.cpp
  DriverUtils.cpp
  EhFrame.cpp
  Error.cpp
  Filesystem.cpp
  GdbIndex.cpp
  ICF.cpp
  InputFiles.cpp
  InputSection.cpp
  LTO.cpp
  LinkerScript.cpp
  MapFile.cpp
  MarkLive.cpp
  Mips.cpp
  OutputSections.cpp
  Relocations.cpp
  ScriptLexer.cpp
  ScriptParser.cpp
  Strings.cpp
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
  ${LLVM_PTHREAD_LIB}

  DEPENDS
  ELFOptionsTableGen
  ${tablegen_deps}
  )
