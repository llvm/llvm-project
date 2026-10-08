# Not add_clang_library: this is not part of clang's public library interface,
# the sources are shared between offload-arch and its unit tests.
add_llvm_library(clangOffloadArch STATIC
  AMDGPUArchByHIP.cpp
  AMDGPUArchByKFD.cpp
  LevelZeroArch.cpp
  NVPTXArch.cpp

  BUILDTREE_ONLY

  LINK_COMPONENTS
  Support
  TargetParser
  )
