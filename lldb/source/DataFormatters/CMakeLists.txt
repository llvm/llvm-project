set(LLVM_NO_RTTI 1)

add_lldb_library(lldbDataFormatters
  CF.cpp
  Cocoa.cpp
  CXXFormatterFunctions.cpp
  DataVisualization.cpp
  FormatCache.cpp
  FormatClasses.cpp
  FormatManager.cpp
  LibCxx.cpp
  LibCxxList.cpp
  LibCxxMap.cpp
  LibStdcpp.cpp
  NSArray.cpp
  NSDictionary.cpp
  NSSet.cpp
  TypeCategory.cpp
  TypeCategoryMap.cpp
  TypeFormat.cpp
  TypeSummary.cpp
  TypeSynthetic.cpp
  )
