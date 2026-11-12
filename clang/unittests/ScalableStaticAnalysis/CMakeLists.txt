add_distinct_clang_unittest(ClangScalableAnalysisTests
  Analyses/EntityPointerLevel/EntityPointerLevelTest.cpp
  Analyses/PointerFlow/PointerFlowTest.cpp
  Analyses/PointerFlow/PointerFlowWPATest.cpp
  Analyses/CallGraph/CallGraphExtractorTest.cpp
  Analyses/TypeConstrainedPointers/TypeConstrainedPointersExtractorTest.cpp
  Analyses/SharedLexicalRepresentation/EntitySourceLocationExtractorTest.cpp
  Analyses/UnsafeBufferUsage/UnsafeBufferUsageTest.cpp
  Analyses/UnsafeBufferUsage/UnsafeBufferUsageWPATest.cpp
  Analyses/VirtualMethodFamily/VirtualMethodFamilyAnalysisTest.cpp
  Analyses/VirtualMethodFamily/VirtualMethodFamilyExtractorTest.cpp
  ASTEntityMappingTest.cpp
  BuildNamespaceTest.cpp
  EntityIdTableTest.cpp
  EntityIdTest.cpp
  EntityLinkageTest.cpp
  EntityLinkerTest.cpp
  EntityNameTest.cpp
  ErrorBuilderTest.cpp
  Frontend/TUSummaryExtractorFrontendActionTest.cpp
  LUSummaryTest.cpp
  ModelStringConversionsTest.cpp
  Registries/FancyAnalysisData.cpp
  Registries/MockSerializationFormat.cpp
  Registries/MockSummaryExtractor1.cpp
  Registries/MockSummaryExtractor2.cpp
  Registries/SerializationFormatRegistryTest.cpp
  Registries/SummaryExtractorRegistryTest.cpp
  Serialization/JSONFormatTest/JSONFormatTest.cpp
  Serialization/JSONFormatTest/LUSummaryTest.cpp
  Serialization/JSONFormatTest/SharedLexicalRepresentationFormatTest.cpp
  Serialization/JSONFormatTest/TUSummaryTest.cpp
  Serialization/JSONFormatTest/VirtualMethodFamilyFormatTest.cpp
  SourceTransformation/CppBoundedBuffersTest.cpp
  SourceTransformation/EmitterTest.cpp
  SourceTransformation/RegistryTest.cpp
  SourceTransformation/SARIFFormatTest.cpp
  SourceTransformation/YAMLFormatTest.cpp
  SummaryData/SummaryDataTest.cpp
  SummaryNameTest.cpp
  TestFixture.cpp
  TUSummaryBuilderTest.cpp
  WholeProgramAnalysis/AnalysisDriverTest.cpp
  WholeProgramAnalysis/UnsafeBufferReachableAnalysisTest.cpp

  CLANG_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangScalableStaticAnalysisAnalyses
  clangScalableStaticAnalysisCore
  clangScalableStaticAnalysisFrontend
  clangScalableStaticAnalysisSourceTransformation
  clangSerialization
  clangTooling
  clangToolingCore

  LINK_LIBS
  LLVMTestingSupport

  LLVM_COMPONENTS
  FrontendOpenMP
  Support
  TargetParser
  )

add_custom_target(check-clang-scalablestaticanalysis-unit
  COMMAND ClangScalableAnalysisTests
  DEPENDS ClangScalableAnalysisTests
  COMMENT "Running the Scalable Static Analysis Framework unit tests"
  USES_TERMINAL
  )
set_target_properties(check-clang-scalablestaticanalysis-unit
  PROPERTIES FOLDER "Clang tests/Suites")
