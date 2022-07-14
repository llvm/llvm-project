set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

set(CLANG_TIDY_CONFUSABLE_CHARS_GEN "clang-tidy-confusable-chars-gen" CACHE
  STRING "Host clang-tidy-confusable-chars-gen executable. Saves building if cross-compiling.")

if(NOT CLANG_TIDY_CONFUSABLE_CHARS_GEN STREQUAL "clang-tidy-confusable-chars-gen")
  set(make_confusable_table ${CLANG_TIDY_CONFUSABLE_CHARS_GEN})
  set(make_confusable_table_target ${CLANG_TIDY_CONFUSABLE_CHARS_GEN})
elseif(LLVM_USE_HOST_TOOLS)
  build_native_tool(clang-tidy-confusable-chars-gen make_confusable_table)
  set(make_confusable_table_target "${make_confusable_table}")
else()
  set(make_confusable_table $<TARGET_FILE:clang-tidy-confusable-chars-gen>)
  set(make_confusable_table_target clang-tidy-confusable-chars-gen)
endif()

add_subdirectory(ConfusableTable)


add_custom_command(
    OUTPUT Confusables.inc
    COMMAND ${make_confusable_table} ${CMAKE_CURRENT_SOURCE_DIR}/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
    DEPENDS ${make_confusable_table_target} ConfusableTable/confusables.txt)

add_custom_target(genconfusable DEPENDS Confusables.inc)

add_clang_library(clangTidyMiscModule
  ConstCorrectnessCheck.cpp
  DefinitionsInHeadersCheck.cpp
  ConfusableIdentifierCheck.cpp
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

  LINK_LIBS
  clangAnalysis
  clangTidy
  clangTidyUtils

  DEPENDS
  omp_gen
  genconfusable
  )

clang_target_link_libraries(clangTidyMiscModule
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangSerialization
  clangTooling
  )
