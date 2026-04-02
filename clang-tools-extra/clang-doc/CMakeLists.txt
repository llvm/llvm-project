set(LLVM_LINK_COMPONENTS
  support
  BitstreamReader
  FrontendOpenMP
  )
add_subdirectory(support)

# Representation.cpp and YAMLGenerator.cpp have conflicting static declarations
# and CommentKind ambiguity when combined in unity builds.
set_source_files_properties(
  Representation.cpp
  YAMLGenerator.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangDoc STATIC
  BitcodeReader.cpp
  BitcodeWriter.cpp
  ClangDoc.cpp
  Generators.cpp
  HTMLGenerator.cpp
  Mapper.cpp
  MDGenerator.cpp
  Representation.cpp
  Serialize.cpp
  YAMLGenerator.cpp
  JSONGenerator.cpp
  MDMustacheGenerator.cpp

  DEPENDS
  omp_gen
  ClangDriverOptions
)

clang_target_link_libraries(clangDoc
  PRIVATE
  clangDocSupport
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangTooling
  clangToolingCore
  clangUnifiedSymbolResolution
  )

target_link_libraries(clangDoc
  PRIVATE
  clangDocSupport
  )

add_subdirectory(tool)

if (LLVM_INCLUDE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
