# We only need to build clang-tblgen if we haven't been told to use an existing binary.
if (NOT DEFINED CLANG_TABLEGEN_EXE)
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
    MveEmitter.cpp
    NeonEmitter.cpp
    RISCVVEmitter.cpp
    SveEmitter.cpp
    TableGen.cpp
    )

  target_link_libraries(clang-tblgen PRIVATE clangSupport_tablegen)
endif()
