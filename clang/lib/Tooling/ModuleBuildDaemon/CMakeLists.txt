set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangModuleBuildDaemon
  Client.cpp
  SocketSupport.cpp
  Utils.cpp
  )
