set(LLVM_LINK_COMPONENTS Support)

add_tablegen(clang-tblgen CLANG
  DESTINATION "${CLANG_TOOLS_INSTALL_DIR}"
  EXPORT Clang
  ASTTableGen.cpp
  CIRLoweringEmitter.cpp
  ClangASTNodesEmitter.cpp
  ClangASTPropertiesEmitter.cpp
  ClangAttrEmitter.cpp
  ClangBuiltinsEmitter.cpp
  ClangBuiltinTemplatesEmitter.cpp
  ClangCommentCommandInfoEmitter.cpp
  ClangCommentHTMLNamedCharacterReferenceEmitter.cpp
  ClangCommentHTMLTagsEmitter.cpp
  ClangDataCollectorsEmitter.cpp
  ClangDiagnosticsEmitter.cpp
  ClangOpcodesEmitter.cpp
  ClangOpenCLBuiltinEmitter.cpp
  ClangOptionDocEmitter.cpp
  ClangSACheckersEmitter.cpp
  ClangSyntaxEmitter.cpp
  ClangTypeNodesEmitter.cpp
  HLSLEmitter.cpp
  MveEmitter.cpp
  NeonEmitter.cpp
  RISCVVEmitter.cpp
  SveEmitter.cpp
  TableGen.cpp
  )

# MveEmitter.cpp and NeonEmitter.cpp define large anonymous-namespace classes
# that conflict with each other in unity builds.
set_source_files_properties(
  NeonEmitter.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

target_link_libraries(clang-tblgen PRIVATE clangSupport_tablegen)
