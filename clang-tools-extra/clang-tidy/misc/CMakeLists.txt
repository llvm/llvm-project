set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

set(CLANG_TIDY_CONFUSABLE_CHARS_GEN "clang-tidy-confusable-chars-gen" CACHE
  STRING "Host clang-tidy-confusable-chars-gen executable. Saves building if cross-compiling.")

if(NOT CLANG_TIDY_CONFUSABLE_CHARS_GEN STREQUAL "clang-tidy-confusable-chars-gen")
  set(clang_tidy_confusable_chars_gen ${CLANG_TIDY_CONFUSABLE_CHARS_GEN})
  set(clang_tidy_confusable_chars_gen_target ${CLANG_TIDY_CONFUSABLE_CHARS_GEN})
elseif(LLVM_USE_HOST_TOOLS)
  build_native_tool(clang-tidy-confusable-chars-gen clang_tidy_confusable_chars_gen)
  set(clang_tidy_confusable_chars_gen_target "${clang_tidy_confusable_chars_gen}")
else()
  set(clang_tidy_confusable_chars_gen $<TARGET_FILE:clang-tidy-confusable-chars-gen>)
  set(clang_tidy_confusable_chars_gen_target clang-tidy-confusable-chars-gen)
endif()

add_subdirectory(ConfusableTable)


add_custom_command(
    OUTPUT Confusables.inc
    COMMAND ${clang_tidy_confusable_chars_gen} ${CMAKE_CURRENT_SOURCE_DIR}/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
    DEPENDS ${clang_tidy_confusable_chars_gen_target} ConfusableTable/confusables.txt)

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
