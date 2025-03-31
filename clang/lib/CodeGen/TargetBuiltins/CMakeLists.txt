include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_library(clangCodeGenTargetBuiltins STATIC
  ARM.cpp
  AMDGPU.cpp
  Hexagon.cpp
  NVPTX.cpp
  PPC.cpp
  RISCV.cpp
  SPIR.cpp
  SystemZ.cpp
  WebAssembly.cpp
  X86.cpp
)
