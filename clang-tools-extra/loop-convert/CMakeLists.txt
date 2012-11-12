set(LLVM_LINK_COMPONENTS support)
set(LLVM_USED_LIBS clangTooling clangBasic clangAST)

add_clang_executable(loop-convert
  LoopConvert.cpp
  LoopActions.cpp
  LoopActions.h
  LoopMatchers.cpp
  LoopMatchers.h
  StmtAncestor.cpp
  StmtAncestor.h
  VariableNaming.cpp
  VariableNaming.h
  )

target_link_libraries(loop-convert
  clangTooling
  clangBasic
  clangASTMatchers
  )
