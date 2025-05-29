add_clang_library(clangABI
  QualTypeMapper.cpp

  LINK_LIBS
  clangAST
  clangBasic
  LLVMABI
  LLVMSupport
)
