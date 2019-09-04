set(LLVM_LINK_COMPONENTS
  )

clang_tablegen(Opcodes.inc
  -gen-clang-opcodes
  SOURCE Opcodes.td
  TARGET Opcodes)

add_clang_library(clangInterp
  Block.cpp
  ByteCodeEmitter.cpp
  ByteCodeExprGen.cpp
  ByteCodeGenError.cpp
  ByteCodeStmtGen.cpp
  Context.cpp
  Descriptor.cpp
  Disasm.cpp
  EvalEmitter.cpp
  Frame.cpp
  Function.cpp
  Interp.cpp
  InterpFrame.cpp
  InterpStack.cpp
  InterpState.cpp
  Pointer.cpp
  Program.cpp
  Record.cpp
  Source.cpp
  State.cpp
  Type.cpp

  DEPENDS
  Opcodes
  )
