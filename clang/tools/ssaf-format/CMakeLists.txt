set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_tool(ssaf-format
  SSAFFormat.cpp
  )

clang_target_link_libraries(ssaf-format
  PRIVATE
  clangAnalysisScalable
  clangBasic
  )
