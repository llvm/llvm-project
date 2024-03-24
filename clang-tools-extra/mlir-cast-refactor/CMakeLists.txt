set(LLVM_LINK_COMPONENTS support)

add_clang_executable(mlirCastRefactor
  MlirCastRefactor.cpp
  )
target_link_libraries(mlirCastRefactor
  PRIVATE
    clangAST
    clangBasic
    clangFormat
    clangFrontend
    clangLex
    clangRewrite
    clangSerialization
    clangTooling
    clangToolingCore
    clangToolingRefactoring
  )