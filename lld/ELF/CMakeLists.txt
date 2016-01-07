set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(ELFOptionsTableGen)

add_lld_library(lldELF2
  Driver.cpp
  DriverUtils.cpp
  Error.cpp
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
  Object
  Option
  MC
  Support
  )

add_dependencies(lldELF2 ELFOptionsTableGen)
