include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_library(clangCodeGenTargets STATIC
  AArch64.cpp
  AMDGPU.cpp
  ARC.cpp
  ARM.cpp
  AVR.cpp
  BPF.cpp
  CSKY.cpp
  DirectX.cpp
  Hexagon.cpp
  Lanai.cpp
  LoongArch.cpp
  M68k.cpp
  MSP430.cpp
  Mips.cpp
  NVPTX.cpp
  PNaCl.cpp
  PPC.cpp
  RISCV.cpp
  SPIR.cpp
  Sparc.cpp
  SystemZ.cpp
  TCE.cpp
  VE.cpp
  WebAssembly.cpp
  X86.cpp
  XCore.cpp
)

target_link_libraries(clangCodeGenTargets
  PRIVATE
  clangCodeGen
)
