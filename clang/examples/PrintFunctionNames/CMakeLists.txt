set(MODULE TRUE)

set( LLVM_LINK_COMPONENTS support mc)

add_clang_library(PrintFunctionNames PrintFunctionNames.cpp)

add_dependencies(PrintFunctionNames
  ClangAttrClasses
  ClangAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangStmtNodes
  )

target_link_libraries(PrintFunctionNames
  clangFrontend
  clangAST
  )

set_target_properties(PrintFunctionNames
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
