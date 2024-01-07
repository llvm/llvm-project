set(LLVM_LINK_COMPONENTS Support)

add_clang_tool(clang-symbolgraph-merger SymbolGraphMergerMain.cpp)

target_link_libraries(clang-symbolgraph-merger PRIVATE
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
)

target_link_libraries(clang-symbolgraph-merger PRIVATE
  clangSymbolGraphMerger)
