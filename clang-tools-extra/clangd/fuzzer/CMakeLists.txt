set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
  )

# This fuzzer runs on oss-fuzz, so keep it around even if it looks unreferenced.
add_llvm_fuzzer(clangd-fuzzer
  clangd-fuzzer.cpp
  DUMMY_MAIN FuzzerClangdMain.cpp
  )

clang_target_link_libraries(clangd-fuzzer
  PRIVATE
  clangBasic
  clangFormat
  clangFrontend
  clangSema
  clangTooling
  clangToolingCore
  )
target_link_libraries(clangd-fuzzer
  PRIVATE
  clangDaemon
  clangdSupport
  )
