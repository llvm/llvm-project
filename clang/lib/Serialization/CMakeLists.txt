set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  Object
  Support
  TargetParser
  )

# ASTWriter.cpp defines HeaderFileInfoTrait and LazySpecializationInfoLookupTrait
# in an anonymous namespace, conflicting with identically-named classes from
# ASTReaderInternals.h included by other files in this directory.
set_source_files_properties(
  ASTWriter.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangSerialization
  ASTCommon.cpp
  ASTReader.cpp
  ASTReaderDecl.cpp
  ASTReaderStmt.cpp
  ASTWriter.cpp
  ASTWriterDecl.cpp
  ASTWriterStmt.cpp
  GeneratePCH.cpp
  GlobalModuleIndex.cpp
  InMemoryModuleCache.cpp
  ModuleCache.cpp
  ModuleFile.cpp
  ModuleFileExtension.cpp
  ModuleManager.cpp
  PCHContainerOperations.cpp
  ObjectFilePCHContainerReader.cpp
  TemplateArgumentHasher.cpp

  ADDITIONAL_HEADERS
  ASTCommon.h
  ASTReaderInternals.h

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  clangSema

  DEPENDS
  omp_gen
  ClangDriverOptions
  )
