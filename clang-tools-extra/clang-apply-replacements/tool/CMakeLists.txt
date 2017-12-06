set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(clang-apply-replacements
  ClangApplyReplacementsMain.cpp
  )
target_link_libraries(clang-apply-replacements
  PRIVATE
  clangApplyReplacements
  clangBasic
  clangFormat
  clangRewrite
  clangToolingCore
  )

install(TARGETS clang-apply-replacements
  RUNTIME DESTINATION bin)
