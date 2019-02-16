include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

set(LLVM_LINK_COMPONENTS
    Support
    )

add_clang_executable(clangd-indexer
  IndexerMain.cpp
  )

target_link_libraries(clangd-indexer
  PRIVATE
  clangAST
  clangIndex
  clangDaemon
  clangBasic
  clangFrontend
  clangLex
  clangTooling
)
