set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(clang-apply-replacements
  ClangApplyReplacementsMain.cpp
  )
target_link_libraries(clang-apply-replacements
  PRIVATE
  clangApplyReplacements
  clangBasic
  clangFormat
  clangRewrite
  clangToolingCore
  clangToolingRefactoring
  )

install(TARGETS clang-apply-replacements
  RUNTIME DESTINATION bin)
