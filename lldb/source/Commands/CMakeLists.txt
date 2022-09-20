lldb_tablegen(CommandOptions.inc -gen-lldb-option-defs
  SOURCE Options.td
  TARGET LLDBOptionsGen)

if (LLDB_ENABLE_SWIFT_SUPPORT)
  set(Healtcheck_SOURCES CommandObjectHealthcheck.cpp)
endif()
set(LLVM_OPTIONAL_SOURCES CommandObjectHealthcheck.cpp)

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
  CommandObjectMemoryTag.cpp
  CommandObjectMultiword.cpp
  CommandObjectPlatform.cpp
  CommandObjectPlugin.cpp
  CommandObjectProcess.cpp
  CommandObjectQuit.cpp
  CommandObjectRegexCommand.cpp
  CommandObjectRegister.cpp
  CommandObjectScript.cpp
  CommandObjectSession.cpp
  CommandObjectSettings.cpp
  CommandObjectSource.cpp
  CommandObjectStats.cpp
  CommandObjectTarget.cpp
  CommandObjectThread.cpp
  CommandObjectThreadUtil.cpp
  CommandObjectTrace.cpp
  CommandObjectType.cpp
  CommandObjectVersion.cpp
  CommandObjectWatchpoint.cpp
  CommandObjectWatchpointCommand.cpp
  CommandOptionArgumentTable.cpp
  CommandOptionsProcessLaunch.cpp

  ${Healtcheck_SOURCES}

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbVersion

  LINK_COMPONENTS
    Support
  )

add_dependencies(lldbCommands LLDBOptionsGen)
