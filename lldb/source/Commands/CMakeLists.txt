lldb_tablegen(CommandOptions.inc -gen-lldb-option-defs
  SOURCE Options.td
  TARGET LLDBOptionsGen)

add_lldb_library(lldbCommands
  CommandCompletions.cpp
  CommandObjectApropos.cpp
  CommandObjectBreakpoint.cpp
  CommandObjectBreakpointCommand.cpp
  CommandObjectCommands.cpp
  CommandObjectDisassemble.cpp
  CommandObjectExpression.cpp
  CommandObjectFrame.cpp
  CommandObjectGUI.cpp
  CommandObjectHelp.cpp
  CommandObjectLanguage.cpp
  CommandObjectLog.cpp
  CommandObjectMemory.cpp
  CommandObjectMultiword.cpp
  CommandObjectPlatform.cpp
  CommandObjectPlugin.cpp
  CommandObjectProcess.cpp
  CommandObjectQuit.cpp
  CommandObjectRegister.cpp
  CommandObjectReproducer.cpp
  CommandObjectSession.cpp
  CommandObjectSettings.cpp
  CommandObjectSource.cpp
  CommandObjectStats.cpp
  CommandObjectTarget.cpp
  CommandObjectThread.cpp
  CommandObjectType.cpp
  CommandObjectVersion.cpp
  CommandObjectWatchpoint.cpp
  CommandObjectWatchpointCommand.cpp

  LINK_LIBS
    lldbBase
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility

  LINK_COMPONENTS
    Support
  )

add_dependencies(lldbCommands LLDBOptionsGen)
