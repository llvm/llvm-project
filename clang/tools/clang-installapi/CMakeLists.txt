set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  TextAPI
  )

add_clang_tool(clang-installapi
  ClangInstallAPI.cpp
  Options.cpp

  GENERATE_DRIVER
  )

clang_target_link_libraries(clang-installapi
  PRIVATE
  clangInstallAPI
  clangBasic
  clangDriver
  clangFrontend
  clangTooling
  )
