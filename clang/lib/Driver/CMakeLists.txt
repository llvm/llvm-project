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
  Compilation.cpp
  CrossWindowsToolChain.cpp
  Driver.cpp
  DriverOptions.cpp
  Job.cpp
  MinGWToolChain.cpp
  Multilib.cpp
  MSVCToolChain.cpp
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
