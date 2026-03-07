set(LLVM_LINK_COMPONENTS
  LineEditor
  Support
  )

add_clang_executable(dexp
  Dexp.cpp

  # Disable PCH, gRPC adds -pthread.
  DISABLE_PCH_REUSE
  )

clang_target_link_libraries(dexp
  PRIVATE
  clangBasic
  )
target_link_libraries(dexp
  PRIVATE
  clangDaemon
  clangdRemoteIndex
  )
