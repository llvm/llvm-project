# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES ClangdToolMain.cpp)

add_clang_library(clangdMain
  ClangdMain.cpp
  Check.cpp
  )

add_clang_tool(clangd
  ClangdToolMain.cpp
  $<TARGET_OBJECTS:obj.clangDaemonTweaks>
  )

set(LLVM_LINK_COMPONENTS
  support
  )

set(CLANGD_XPC_LIBS "")
if(CLANGD_BUILD_XPC)
  list(APPEND CLANGD_XPC_LIBS "clangdXpcJsonConversions" "clangdXpcTransport")
endif()

clang_target_link_libraries(clangdMain
  PRIVATE
  clangAST
  clangBasic
  clangFormat
  clangFrontend
  clangTooling
  clangToolingSyntax
  )

target_link_libraries(clangdMain
  PRIVATE
  clangTidy
  clangTidyUtils

  clangDaemon
  clangdRemoteIndex
  clangdSupport
  ${CLANGD_XPC_LIBS}
  )

clang_target_link_libraries(clangd
  PRIVATE
  clangAST
  clangBasic
  clangLex
  clangSema
  clangToolingCore
  clangToolingRefactoring
  clangToolingSyntax
  )

target_link_libraries(clangd
  PRIVATE
  clangdMain
  clangDaemon
  clangdSupport
  )
