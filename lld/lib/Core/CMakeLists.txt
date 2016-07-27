add_lld_library(lldCore
  DefinedAtom.cpp
  Error.cpp
  File.cpp
  LinkingContext.cpp
  Reader.cpp
  Reproduce.cpp
  Resolver.cpp
  SymbolTable.cpp
  Writer.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLD_INCLUDE_DIR}/lld/Core

  LINK_LIBS
    LLVMSupport
  )
