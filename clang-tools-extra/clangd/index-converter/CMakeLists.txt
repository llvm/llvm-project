set(LLVM_LINK_COMPONENTS
    Support
    )

add_clang_executable(clangd-index-converter
  IndexConverterMain.cpp
  )

clang_target_link_libraries(clangd-index-converter
  PRIVATE
  clangBasic
  )
target_link_libraries(clangd-index-converter
  PRIVATE
  clangDaemon
  clangdSupport
  )
