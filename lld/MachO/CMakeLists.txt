set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(MachOOptionsTableGen)

add_lld_library(lldMachO2
  Arch/X86_64.cpp
  Driver.cpp
  DriverUtils.cpp
  ExportTrie.cpp
  InputFiles.cpp
  InputSection.cpp
  MergedOutputSection.cpp
  ObjC.cpp
  OutputSection.cpp
  OutputSegment.cpp
  SymbolTable.cpp
  Symbols.cpp
  SyntheticSections.cpp
  Target.cpp
  Writer.cpp

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BinaryFormat
  Core
  Object
  Option
  Support
  TextAPI

  LINK_LIBS
  lldCommon
  ${LLVM_PTHREAD_LIB}

  DEPENDS
  MachOOptionsTableGen
  ${tablegen_deps}
  )
