set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyCppCoreGuidelinesModule
  CppCoreGuidelinesTidyModule.cpp
  ProBoundsArrayToPointerDecayCheck.cpp
  ProBoundsConstantArrayIndexCheck.cpp
  ProBoundsPointerArithmeticCheck.cpp
  ProTypeConstCastCheck.cpp
  ProTypeCstyleCastCheck.cpp
  ProTypeReinterpretCastCheck.cpp
  ProTypeStaticCastDowncastCheck.cpp
  ProTypeUnionAccessCheck.cpp
  ProTypeVarargCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyMiscModule
  clangTidyUtils
  clangTooling
  )
