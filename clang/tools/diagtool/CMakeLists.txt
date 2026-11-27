set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(DiagtoolOptsTableGen)

add_clang_tool(diagtool
  diagtool_main.cpp
  DiagTool.cpp
  DiagnosticNames.cpp
  FindDiagnosticID.cpp
  ListWarnings.cpp
  ShowEnabledWarnings.cpp
  TreeView.cpp

  DEPENDS
  DiagtoolOptsTableGen
)

clang_target_link_libraries(diagtool
  PRIVATE
  clangBasic
  clangFrontend
  )
