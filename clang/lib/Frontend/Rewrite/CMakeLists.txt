set(LLVM_LINK_COMPONENTS
  Support
  )

# Both files include config.h which uses #error to prevent double-inclusion,
# breaking unity builds when they end up in the same translation unit.
set_source_files_properties(
  RewriteModernObjC.cpp
  RewriteObjC.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangRewriteFrontend
  FixItRewriter.cpp
  FrontendActions.cpp
  HTMLPrint.cpp
  InclusionRewriter.cpp
  RewriteMacros.cpp
  RewriteModernObjC.cpp
  RewriteObjC.cpp
  RewriteTest.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangEdit
  clangFrontend
  clangLex
  clangRewrite
  clangSerialization

  DEPENDS
  omp_gen
  )
