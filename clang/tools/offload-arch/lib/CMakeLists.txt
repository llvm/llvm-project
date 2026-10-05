# Not add_clang_library: this is not part of clang's public library interface,
# the sources are shared between offload-arch and its unit tests. Use
# llvm_add_library directly, because add_llvm_library ignores STATIC and builds
# a shared library with BUILD_SHARED_LIBS=ON, which is never installed.
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
