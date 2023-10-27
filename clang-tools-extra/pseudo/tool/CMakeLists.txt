set(LLVM_LINK_COMPONENTS support)

add_clang_tool(clang-pseudo
  ClangPseudo.cpp
  HTMLForest.cpp
  )

clang_target_link_libraries(clang-pseudo
  PRIVATE
  clangBasic
  )

target_link_libraries(clang-pseudo
  PRIVATE
  clangPseudo
  clangPseudoGrammar
  clangPseudoCLI
  )

add_custom_command(OUTPUT HTMLForestResources.inc
  COMMAND "${Python3_EXECUTABLE}" ${CLANG_SOURCE_DIR}/utils/bundle_resources.py
  ${CMAKE_CURRENT_BINARY_DIR}/HTMLForestResources.inc
  HTMLForest.css HTMLForest.js HTMLForest.html
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Bundling HTMLForest resources"
  DEPENDS ${CLANG_SOURCE_DIR}/utils/bundle_resources.py HTMLForest.css HTMLForest.js HTMLForest.html
  VERBATIM)
add_custom_target(clang-pseudo-resources DEPENDS HTMLForestResources.inc)
add_dependencies(clang-pseudo clang-pseudo-resources)
