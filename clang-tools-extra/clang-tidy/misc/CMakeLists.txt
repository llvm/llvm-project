set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

if(LLVM_USE_HOST_TOOLS)
  build_native_tool(make-confusable-table make_confusable_table)
  set(make_confusable_table_target "${make_confusable_table}")
else()
  set(make_confusable_table $<TARGET_FILE:make-confusable-table>)
  set(make_confusable_table_target make-confusable-table)
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
