set(LLVM_LINK_COMPONENTS
  Support
  )

# TreeView.cpp defines printUsage which conflicts with identically-named
# functions from other diagtool files in unity builds.
set_source_files_properties(
  TreeView.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_tool(diagtool
  diagtool_main.cpp
  DiagTool.cpp
  DiagnosticNames.cpp
  FindDiagnosticID.cpp
  ListWarnings.cpp
  ShowEnabledWarnings.cpp
  TreeView.cpp
)

clang_target_link_libraries(diagtool
  PRIVATE
  clangBasic
  clangDriver
  clangFrontend
  )
