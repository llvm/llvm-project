set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangAnalysisScalable
  ASTEntityMapping.cpp
  EntityLinker/EntityLinker.cpp
  Model/BuildNamespace.cpp
  Model/EntityIdTable.cpp
  Model/EntityName.cpp
  Serialization/JSONFormat.cpp
  Serialization/SerializationFormatRegistry.cpp
  Support/ErrorBuilder.cpp
  TUSummary/ExtractorRegistry.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangIndex
  clangLex
  clangFrontend

  DEPENDS
  )
