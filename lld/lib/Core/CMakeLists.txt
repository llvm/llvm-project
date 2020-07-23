add_lld_library(lldCore
  DefinedAtom.cpp
  Error.cpp
  File.cpp
  LinkingContext.cpp
  Reader.cpp
  Resolver.cpp
  SymbolTable.cpp
  Writer.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLD_INCLUDE_DIR}/lld/Core

  LINK_COMPONENTS
    BinaryFormat
    MC
    Support

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  DEPENDS
  intrinsics_gen
  )
