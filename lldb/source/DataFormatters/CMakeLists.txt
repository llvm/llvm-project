add_lldb_library(lldbDataFormatters
  CXXFunctionPointer.cpp
  DataVisualization.cpp
  DumpValueObjectOptions.cpp
  FormatCache.cpp
  FormatClasses.cpp
  FormatManager.cpp
  FormattersHelpers.cpp
  LanguageCategory.cpp
  StringPrinter.cpp
  TypeCategory.cpp
  TypeCategoryMap.cpp
  TypeFormat.cpp
  TypeSummary.cpp
  TypeSynthetic.cpp
  TypeValidator.cpp
  ValueObjectPrinter.cpp
  VectorType.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
  
  LINK_COMPONENTS
    Support
  )

include(AddLLVM)
include(SwiftAddCustomCommandTarget)

set(lib_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX}")

install(
  CODE "file(MAKE_DIRECTORY ${lib_dir}/lldb)")

