set(CLANGD_XPC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CLANGD_XPC_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(CreateClangdXPCFramework)

add_subdirectory(framework)
add_subdirectory(test-client)

set(LLVM_LINK_COMPONENTS
  Support
  )

# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES Conversion.cpp XPCTransport.cpp)

add_clang_library(clangdXpcJsonConversions STATIC
  Conversion.cpp
  LINK_LIBS clangDaemon clangdSupport
  )

add_clang_library(clangdXpcTransport STATIC
  XPCTransport.cpp
  LINK_LIBS clangDaemon clangdSupport clangdXpcJsonConversions
  DEPENDS ClangDriverOptions
  )
