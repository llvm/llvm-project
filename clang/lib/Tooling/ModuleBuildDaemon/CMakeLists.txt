set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangModuleBuildDaemon
  Frontend.cpp
  SocketSupport.cpp
  Utils.cpp
  )
