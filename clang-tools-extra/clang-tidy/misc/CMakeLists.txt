set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

setup_host_tool(clang-tidy-confusable-chars-gen CLANG_TIDY_CONFUSABLE_CHARS_GEN clang_tidy_confusable_chars_gen clang_tidy_confusable_chars_gen_target)

add_subdirectory(ConfusableTable)

include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../include-cleaner/include")

add_custom_command(
    OUTPUT Confusables.inc
    COMMAND ${clang_tidy_confusable_chars_gen} ${CMAKE_CURRENT_SOURCE_DIR}/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
    DEPENDS ${clang_tidy_confusable_chars_gen_target} ConfusableTable/confusables.txt)

add_custom_target(genconfusable DEPENDS Confusables.inc)
set_target_properties(genconfusable PROPERTIES FOLDER "Clang Tools Extra/Sourcegenning")

add_clang_library(clangTidyMiscModule
  ConstCorrectnessCheck.cpp
  CoroutineHostileRAIICheck.cpp
  DefinitionsInHeadersCheck.cpp
  ConfusableIdentifierCheck.cpp
  HeaderIncludeCycleCheck.cpp
  IncludeCleanerCheck.cpp
  MiscTidyModule.cpp
  MisleadingBidirectional.cpp
  MisleadingIdentifier.cpp
  MisplacedConstCheck.cpp
  NewDeleteOverloadsCheck.cpp
  NoRecursionCheck.cpp
  NonCopyableObjects.cpp
  NonPrivateMemberVariablesInClassesCheck.cpp
  RedundantExpressionCheck.cpp
  StaticAssertCheck.cpp
  ThrowByValueCatchByReferenceCheck.cpp
  UnconventionalAssignOperatorCheck.cpp
  UniqueptrResetReleaseCheck.cpp
  UnusedAliasDeclsCheck.cpp
  UnusedParametersCheck.cpp
  UnusedUsingDeclsCheck.cpp
  UseAnonymousNamespaceCheck.cpp

  LINK_LIBS
  clangAnalysis
  clangTidy
  clangTidyUtils

  DEPENDS
  omp_gen
  genconfusable
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyMiscModule
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangLex
  clangSerialization
  clangTooling
  clangToolingInclusions
  clangToolingInclusionsStdlib
  )
target_link_libraries(clangTidyMiscModule
  PRIVATE
  clangIncludeCleaner
  )
