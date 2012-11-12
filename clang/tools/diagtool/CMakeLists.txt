set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

add_clang_executable(diagtool
  diagtool_main.cpp
  DiagTool.cpp
  DiagnosticNames.cpp
  ListWarnings.cpp
  ShowEnabledWarnings.cpp
  TreeView.cpp
)

add_dependencies(diagtool
  ClangDiagnosticIndexName
  )

target_link_libraries(diagtool
  clangBasic
  clangLex
  clangSema
  clangFrontend
  )

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
else()
  set(CLANGXX_LINK_OR_COPY copy)
endif()
