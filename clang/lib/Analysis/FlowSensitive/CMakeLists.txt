add_clang_library(clangAnalysisFlowSensitive
  Arena.cpp
  ControlFlowContext.cpp
  DataflowAnalysisContext.cpp
  DataflowEnvironment.cpp
  HTMLLogger.cpp
  Logger.cpp
  Transfer.cpp
  TypeErasedDataflowAnalysis.cpp
  Value.cpp
  WatchedLiteralsSolver.cpp
  DebugSupport.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangBasic
  clangLex
  )

add_subdirectory(Models)

add_custom_command(OUTPUT HTMLLogger.inc
  COMMAND "${Python3_EXECUTABLE}" ${CLANG_SOURCE_DIR}/utils/bundle_resources.py 
  ${CMAKE_CURRENT_BINARY_DIR}/HTMLLogger.inc
  HTMLLogger.html HTMLLogger.css HTMLLogger.js
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Bundling HTMLLogger resources"
  DEPENDS ${CLANG_SOURCE_DIR}/utils/bundle_resources.py HTMLLogger.html HTMLLogger.css HTMLLogger.js
  VERBATIM)
add_custom_target(clangAnalysisFlowSensitiveResources DEPENDS HTMLLogger.inc)
add_dependencies(clangAnalysisFlowSensitive clangAnalysisFlowSensitiveResources)
