set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_tool(ssaf-linker
  SSAFLinker.cpp
  )

clang_target_link_libraries(ssaf-linker
  PRIVATE
  clangAnalysisScalable
  clangBasic
  )
