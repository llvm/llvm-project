set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangAnalysisScalable
  ASTEntityMapping.cpp
  EntityLinker/EntityLinker.cpp
  Frontend/TUSummaryExtractorFrontendAction.cpp
  Model/BuildNamespace.cpp
  Model/EntityId.cpp
  Model/EntityIdTable.cpp
  Model/EntityLinkage.cpp
  Model/EntityName.cpp
  Model/SummaryName.cpp
  Serialization/JSONFormat/JSONFormatImpl.cpp
  Serialization/JSONFormat/LUSummary.cpp
  Serialization/JSONFormat/LUSummaryEncoding.cpp
  Serialization/JSONFormat/TUSummary.cpp
  Serialization/JSONFormat/TUSummaryEncoding.cpp
  Serialization/SerializationFormatRegistry.cpp
  Support/ErrorBuilder.cpp
  TUSummary/ExtractorRegistry.cpp
  TUSummary/TUSummaryBuilder.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangIndex
  clangLex
  clangSema

  DEPENDS
  )
