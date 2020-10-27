set(LLVM_LINK_COMPONENTS
  Analysis
  CodeGen
  Core
  ExecutionEngine
  IPO
  IRReader
  MC
  MCJIT
  Object
  RuntimeDyld
  SelectionDAG
  Support
  Target
  TransformUtils
  native
)

add_clang_library(clangHandleLLVM
  handle_llvm.cpp

  DEPENDS
  intrinsics_gen
  )
