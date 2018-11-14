set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyCppCoreGuidelinesModule
  AvoidGotoCheck.cpp
  CppCoreGuidelinesTidyModule.cpp
  InterfacesGlobalInitCheck.cpp
  MacroUsageCheck.cpp
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
  SlicingCheck.cpp
  SpecialMemberFunctionsCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyMiscModule
  clangTidyModernizeModule
  clangTidyReadabilityModule
  clangTidyUtils
  clangTooling
  )
