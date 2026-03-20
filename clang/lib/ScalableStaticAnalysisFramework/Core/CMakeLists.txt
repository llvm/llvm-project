set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkCore
  ASTEntityMapping.cpp
  EntityLinker/EntityLinker.cpp
  Model/BuildNamespace.cpp
  Model/EntityId.cpp
  Model/EntityIdTable.cpp
  Model/EntityLinkage.cpp
  Model/EntityName.cpp
  Model/SummaryName.cpp
  Serialization/JSONFormat/JSONEntitySummaryEncoding.cpp
  Serialization/JSONFormat/JSONFormatImpl.cpp
  Serialization/JSONFormat/LUSummary.cpp
  Serialization/JSONFormat/LUSummaryEncoding.cpp
  Serialization/JSONFormat/TUSummary.cpp
  Serialization/JSONFormat/TUSummaryEncoding.cpp
  Serialization/SerializationFormatRegistry.cpp
  SummaryData/LUSummaryConsumer.cpp
  SummaryData/SummaryDataBuilderRegistry.cpp
  Support/ErrorBuilder.cpp
  TUSummary/ExtractorRegistry.cpp
  TUSummary/TUSummaryBuilder.cpp
  WholeProgramAnalysis/AnalysisDriver.cpp
  WholeProgramAnalysis/AnalysisName.cpp
  WholeProgramAnalysis/AnalysisRegistry.cpp

  LINK_LIBS
  clangAST
  clangUnifiedSymbolResolution

  DEPENDS
  )
