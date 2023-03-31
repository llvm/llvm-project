set(SWIFT_LIBS lldbPluginExpressionParserSwift)
if(NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_LIBS)
endif()

# TODO: Add property `NO_PLUGIN_DEPENDENCIES` to lldbExpression
add_lldb_library(lldbExpression NO_PLUGIN_DEPENDENCIES
  DiagnosticManager.cpp
  DWARFExpression.cpp
  DWARFExpressionList.cpp
  Expression.cpp
  ExpressionVariable.cpp
  FunctionCaller.cpp
  IRExecutionUnit.cpp
  IRInterpreter.cpp
  IRMemoryMap.cpp
  LLVMUserExpression.cpp
  Materializer.cpp
  ObjectFileJIT.cpp
  REPL.cpp
  UserExpression.cpp
  UtilityFunction.cpp

  DEPENDS
  intrinsics_gen

  LINK_LIBS
    lldbCore
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility

    # Begin Swift Mods
    ${SWIFT_LIBS}
    # End Swift Mods

  LINK_COMPONENTS
    Core
    ExecutionEngine
    Support
  )
