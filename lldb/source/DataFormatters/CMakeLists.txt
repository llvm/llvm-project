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
  )

include(AddLLVM)
include(SwiftAddCustomCommandTarget)

set(lib_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX}")

add_custom_command_target(
    unused_var
    COMMAND "${CMAKE_COMMAND}" "-E" "copy" "${CMAKE_CURRENT_SOURCE_DIR}/DumpForDebugger.swift"
"${lib_dir}/lldb/DumpForDebugger.swift"
    OUTPUT "${lib_dir}/lldb/DumpForDebugger.swift"
    VERBATIM
    ALL
    DEPENDS DumpForDebugger.swift)

install(
  CODE "file(MAKE_DIRECTORY ${lib_dir}/lldb)")

install(
  FILES "${lib_dir}/lldb/DumpForDebugger.swift"
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/lldb/)
