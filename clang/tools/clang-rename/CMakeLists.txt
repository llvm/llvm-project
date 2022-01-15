set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_tool(clang-rename
  ClangRename.cpp
  )

clang_target_link_libraries(clang-rename
  PRIVATE
  clangBasic
  clangFrontend
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  clangToolingRefactoring
  )

install(PROGRAMS clang-rename.py
  DESTINATION share/clang
  COMPONENT clang-rename)
install(PROGRAMS clang-rename.el
  DESTINATION share/clang
  COMPONENT clang-rename)
