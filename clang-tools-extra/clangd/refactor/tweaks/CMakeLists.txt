include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)

set(LLVM_LINK_COMPONENTS
  support
  )

# A target containing all code tweaks (i.e. mini-refactorings) provided by
# clangd.
# Built as an object library to make sure the linker does not remove global
# constructors that register individual tweaks in a global registry.
# To enable these tweaks in executables or shared libraries, add
# $<TARGET_OBJECTS:obj.clangDaemonTweaks> to a list of sources, see
# clangd/tool/CMakeLists.txt for an example.
add_clang_library(clangDaemonTweaks OBJECT
  AddUsing.cpp
  AnnotateHighlightings.cpp
  DumpAST.cpp
  DefineInline.cpp
  DefineOutline.cpp
  ExpandDeducedType.cpp
  ExpandMacro.cpp
  ExtractFunction.cpp
  ExtractVariable.cpp
  MemberwiseConstructor.cpp
  ObjCLocalizeStringLiteral.cpp
  ObjCMemberwiseInitializer.cpp
  PopulateSwitch.cpp
  RawStringLiteral.cpp
  RemoveUsingNamespace.cpp
  ScopifyEnum.cpp
  SpecialMembers.cpp
  SwapIfBranches.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangDaemon
  clangdSupport
  clangFormat
  clangLex
  clangSema
  clangToolingCore
  clangToolingRefactoring
  clangToolingSyntax

  DEPENDS
  omp_gen
  ClangDriverOptions
  )
