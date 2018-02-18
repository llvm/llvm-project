include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

set(LLVM_LINK_COMPONENTS
    Support
    )

add_clang_executable(global-symbol-builder
  GlobalSymbolBuilderMain.cpp
  )

target_link_libraries(global-symbol-builder
  PRIVATE
  clangAST
  clangIndex
  clangDaemon
  clangBasic
  clangFrontend
  clangLex
  clangTooling
)
