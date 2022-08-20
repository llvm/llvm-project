set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangExtractAPI
  API.cpp
  AvailabilityInfo.cpp
  ExtractAPIConsumer.cpp
  DeclarationFragments.cpp
  Serialization/SerializerBase.cpp
  Serialization/SymbolGraphSerializer.cpp
  TypedefUnderlyingTypeResolver.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangIndex
  clangLex
  )
