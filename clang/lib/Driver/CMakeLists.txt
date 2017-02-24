set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

if(WIN32)
  # MSVCToolChain.cpp uses version.dll.
  set(system_libs version)
endif()

add_clang_library(clangDriver
  Action.cpp
  Arch/AArch64.cpp
  Arch/ARM.cpp
  Arch/Mips.cpp
  Arch/PPC.cpp
  Arch/Sparc.cpp
  Arch/SystemZ.cpp
  Arch/X86.cpp
  Compilation.cpp
  CrossWindowsToolChain.cpp
  Distro.cpp
  Driver.cpp
  DriverOptions.cpp
  Job.cpp
  MSVCToolChain.cpp
  MinGWToolChain.cpp
  Multilib.cpp
  Phases.cpp
  SanitizerArgs.cpp
  Tool.cpp
  ToolChain.cpp
  ToolChains.cpp
  Tools.cpp
  Types.cpp

  DEPENDS
  ClangDriverOptions

  LINK_LIBS
  clangBasic
  ${system_libs}
  )
