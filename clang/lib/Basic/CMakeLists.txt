set(LLVM_LINK_COMPONENTS mc)

add_clang_library(clangBasic
  Builtins.cpp
  ConvertUTF.c
  ConvertUTFWrapper.cpp
  Diagnostic.cpp
  DiagnosticIDs.cpp
  FileManager.cpp
  FileSystemStatCache.cpp
  IdentifierTable.cpp
  LangOptions.cpp
  Module.cpp
  ObjCRuntime.cpp
  SourceLocation.cpp
  SourceManager.cpp
  TargetInfo.cpp
  Targets.cpp
  TokenKinds.cpp
  Version.cpp
  VersionTuple.cpp
  )

# Determine Subversion revision.
# FIXME: This only gets updated when CMake is run, so this revision number
# may be out-of-date!
if( NOT IS_SYMLINK "${CLANG_SOURCE_DIR}" )  # See PR 8437
  find_package(Subversion)
endif()
if (Subversion_FOUND AND EXISTS "${CLANG_SOURCE_DIR}/.svn")
  Subversion_WC_INFO(${CLANG_SOURCE_DIR} CLANG)
  set_source_files_properties(Version.cpp
    PROPERTIES COMPILE_DEFINITIONS "SVN_REVISION=\"${CLANG_WC_REVISION}\"")
endif()

add_dependencies(clangBasic
  ClangARMNeon
  ClangAttrList
  ClangDiagnosticAnalysis
  ClangDiagnosticAST
  ClangDiagnosticComment
  ClangDiagnosticCommon
  ClangDiagnosticDriver
  ClangDiagnosticFrontend
  ClangDiagnosticGroups
  ClangDiagnosticIndexName
  ClangDiagnosticLex
  ClangDiagnosticParse
  ClangDiagnosticSema
  ClangDiagnosticSerialization
  )
