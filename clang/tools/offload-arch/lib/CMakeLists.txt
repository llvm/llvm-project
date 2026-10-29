# Not add_clang_library: this is not part of clang's public library interface,
# the sources are shared between offload-arch and its unit tests.
llvm_add_library(clangOffloadArch STATIC
  AMDGPUArchByHIP.cpp
  AMDGPUArchByKFD.cpp
  LevelZeroArch.cpp
  NVPTXArch.cpp
  OffloadArchOptions.cpp

  LINK_COMPONENTS
  Support
  TargetParser
  )
