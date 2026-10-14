set(LLVM_LINK_COMPONENTS
    Support
    )

add_clang_executable(clangd-remap
  RemapMain.cpp
  )

clang_target_link_libraries(clangd-remap
  PRIVATE
  clangBasic
  )
target_link_libraries(clangd-remap
  PRIVATE
  clangDaemon
  clangdSupport
  )
