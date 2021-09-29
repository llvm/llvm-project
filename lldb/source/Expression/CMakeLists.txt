set(SWIFT_LIBS lldbPluginExpressionParserSwift)
if(NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_LIBS)
endif()

add_lldb_library(lldbExpression
  DiagnosticManager.cpp
  DWARFExpression.cpp
  Expression.cpp
  ExpressionVariable.cpp
  FunctionCaller.cpp
  IRExecutionUnit.cpp
  IRInterpreter.cpp
  IRMemoryMap.cpp
  LLVMUserExpression.cpp
  Materializer.cpp
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
    lldbPluginObjectFileJIT

    # Begin Swift Mods
    ${SWIFT_LIBS}
    # End Swift Mods

  LINK_COMPONENTS
    Core
    ExecutionEngine
    Support
  )
