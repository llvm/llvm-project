set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
  support
  )

add_clang_tool(clang-tidy
  ClangTidyMain.cpp
  )
add_dependencies(clang-tidy
  clang-resource-headers
  )
target_link_libraries(clang-tidy
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  clangTooling
  clangToolingCore
  ${ALL_CLANG_TIDY_CHECKS}
  )


install(PROGRAMS clang-tidy-diff.py
  DESTINATION share/clang
  COMPONENT clang-tidy)
install(PROGRAMS run-clang-tidy.py
  DESTINATION share/clang
  COMPONENT clang-tidy)
