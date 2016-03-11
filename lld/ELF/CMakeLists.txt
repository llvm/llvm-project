set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(ELFOptionsTableGen)

add_lld_library(lldELF
  Driver.cpp
  DriverUtils.cpp
  Error.cpp
  ICF.cpp
  InputFiles.cpp
  InputSection.cpp
  LinkerScript.cpp
  MarkLive.cpp
  OutputSections.cpp
  SymbolTable.cpp
  Symbols.cpp
  Target.cpp
  Writer.cpp

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BitReader
  BitWriter
  Core
  Linker
  Object
  Option
  MC
  Support
  Target

  LINK_LIBS
  lldConfig
  ${PTHREAD_LIB}
  )

add_dependencies(lldELF ELFOptionsTableGen)
