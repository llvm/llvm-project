set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

set(SOURCES
  ARCMigrate.cpp
  CIndex.cpp
  CIndexCXX.cpp
  CIndexCodeCompletion.cpp
  CIndexDiagnostic.cpp
  CIndexDiagnostic.h
  CIndexHigh.cpp
  CIndexInclusionStack.cpp
  CIndexUSRs.cpp
  CIndexer.cpp
  CIndexer.h
  CXComment.cpp
  CXCursor.cpp
  CXCursor.h
  CXCompilationDatabase.cpp
  CXLoadedDiagnostic.cpp
  CXLoadedDiagnostic.h
  CXSourceLocation.cpp
  CXSourceLocation.h
  CXStoredDiagnostic.cpp
  CXString.cpp
  CXString.h
  CXTranslationUnit.h
  CXType.cpp
  CXType.h
  IndexBody.cpp
  IndexDecl.cpp
  IndexTypeSourceInfo.cpp
  Index_Internal.h
  Indexing.cpp
  IndexingContext.cpp
  IndexingContext.h
  ../../include/clang-c/Index.h
  )

set(LIBRARIES
  clangARCMigrate
  clangRewriteCore
  clangRewriteFrontend
  clangFrontend
  clangDriver
  clangSerialization
  clangSema
  clangEdit
  clangAST
  clangLex
  clangTooling
  clangBasic
  )

set(GENERATED_HEADERS
  ClangAttrClasses
  ClangAttrList
  ClangAttrParsedAttrList
  ClangCommentNodes
  ClangDiagnosticCommon
  ClangDiagnosticFrontend
  ClangDeclNodes
  ClangStmtNodes
  )

if( LLVM_ENABLE_PIC )
  set(SHARED_LIBRARY TRUE)
  add_clang_library(libclang ${SOURCES})
  target_link_libraries(libclang ${LIBRARIES})
  add_dependencies(libclang ${GENERATED_HEADERS})

  if(WIN32)
    set_target_properties(libclang
      PROPERTIES
      OUTPUT_NAME "libclang"
      VERSION ${LIBCLANG_LIBRARY_VERSION}
      DEFINE_SYMBOL _CINDEX_LIB_)
  else()
    set_target_properties(libclang
      PROPERTIES
      OUTPUT_NAME "clang"
      VERSION ${LIBCLANG_LIBRARY_VERSION}
      DEFINE_SYMBOL _CINDEX_LIB_)
  endif()

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(LIBCLANG_LINK_FLAGS
      "-Wl,-compatibility_version -Wl,1 -Wl,-dead_strip")
    set_target_properties(libclang
      PROPERTIES
      LINK_FLAGS "${LIBCLANG_LINK_FLAGS}"
      INSTALL_NAME_DIR "@executable_path/../lib")
  endif()


  set(LIBCLANG_STATIC_TARGET_NAME libclang_static)
else()
  set(LIBCLANG_STATIC_TARGET_NAME libclang)
endif()

if( NOT BUILD_SHARED_LIBS AND NOT WIN32 )
  add_clang_library(${LIBCLANG_STATIC_TARGET_NAME} STATIC ${SOURCES})
  target_link_libraries(${LIBCLANG_STATIC_TARGET_NAME} ${LIBRARIES})
  add_dependencies(${LIBCLANG_STATIC_TARGET_NAME} ${GENERATED_HEADERS})

  set_target_properties(${LIBCLANG_STATIC_TARGET_NAME}
    PROPERTIES
    OUTPUT_NAME "clang")
endif()
