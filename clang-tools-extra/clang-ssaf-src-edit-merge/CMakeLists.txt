set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(clang-ssaf-src-edit-merge
  SrcEditMerge.cpp
  )

# clang-apply-replacements headers live under <peer-tool>/include/ and aren't
# globally exported; expose them only to this tool.
target_include_directories(clang-ssaf-src-edit-merge
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../clang-apply-replacements/include
  )

clang_target_link_libraries(clang-ssaf-src-edit-merge
  PRIVATE
  clangBasic
  clangToolingCore
  clangToolingRefactoring
  )

target_link_libraries(clang-ssaf-src-edit-merge
  PRIVATE
  clangApplyReplacements
  )
