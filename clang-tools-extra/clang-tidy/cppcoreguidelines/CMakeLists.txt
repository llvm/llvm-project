set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyCppCoreGuidelinesModule
  AvoidGotoCheck.cpp
  CppCoreGuidelinesTidyModule.cpp
  InterfacesGlobalInitCheck.cpp
  NarrowingConversionsCheck.cpp
  NoMallocCheck.cpp
  OwningMemoryCheck.cpp
  ProBoundsArrayToPointerDecayCheck.cpp
  ProBoundsConstantArrayIndexCheck.cpp
  ProBoundsPointerArithmeticCheck.cpp
  ProTypeConstCastCheck.cpp
  ProTypeCstyleCastCheck.cpp
  ProTypeMemberInitCheck.cpp
  ProTypeReinterpretCastCheck.cpp
  ProTypeStaticCastDowncastCheck.cpp
  ProTypeUnionAccessCheck.cpp
  ProTypeVarargCheck.cpp
  SpecialMemberFunctionsCheck.cpp
  SlicingCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyMiscModule
  clangTidyReadabilityModule
  clangTidyUtils
  clangTooling
  )
