set(SWIFT_LIBS lldbPluginExpressionParserSwift)
set(PLUGIN_DEPENDENCY_ARG)
if(NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_LIBS)
  set(PLUGIN_DEPENDENCY_ARG NO_PLUGIN_DEPENDENCIES)
endif()

add_lldb_library(lldbExpression ${PLUGIN_DEPENDENCY_ARG}
  DiagnosticManager.cpp
  DWARFExpression.cpp
  DWARFExpressionList.cpp
  Expression.cpp
  ExpressionParser.cpp
  ExpressionTypeSystemHelper.cpp
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
    lldbValueObject

    # Begin Swift Mods
    ${SWIFT_LIBS}
    # End Swift Mods

  LINK_COMPONENTS
    Core
    ExecutionEngine
    Support
  )
