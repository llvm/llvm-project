set(LLDB_CURSES_LIBS)

if (NOT LLDB_DISABLE_CURSES)
  list(APPEND LLDB_CURSES_LIBS ${CURSES_LIBRARIES})
  if(LLVM_ENABLE_TERMINFO AND HAVE_TERMINFO)
    list(APPEND LLDB_CURSES_LIBS ${TERMINFO_LIBS})
  endif()
endif()

add_lldb_library(lldbCore
  Address.cpp
  AddressRange.cpp
  AddressResolver.cpp
  AddressResolverFileLine.cpp
  AddressResolverName.cpp
  Broadcaster.cpp
  Communication.cpp
  Debugger.cpp
  Disassembler.cpp
  DumpDataExtractor.cpp
  DumpRegisterValue.cpp
  DynamicLoader.cpp
  EmulateInstruction.cpp
  Event.cpp
  FileLineResolver.cpp
  FileSpecList.cpp
  FormatEntity.cpp
  IOHandler.cpp
  Listener.cpp
  Mangled.cpp
  Module.cpp
  ModuleChild.cpp
  ModuleList.cpp
  Opcode.cpp
  PluginManager.cpp
  RegisterValue.cpp
  Scalar.cpp
  SearchFilter.cpp
  Section.cpp
  SourceManager.cpp
  State.cpp
  StreamAsynchronousIO.cpp
  StreamFile.cpp
  UserSettingsController.cpp
  Value.cpp
  ValueObject.cpp
  ValueObjectCast.cpp
  ValueObjectChild.cpp
  ValueObjectConstResult.cpp
  ValueObjectConstResultCast.cpp
  ValueObjectConstResultChild.cpp
  ValueObjectConstResultImpl.cpp
  ValueObjectDynamicValue.cpp
  ValueObjectList.cpp
  ValueObjectMemory.cpp
  ValueObjectRegister.cpp
  ValueObjectSyntheticFilter.cpp
  ValueObjectVariable.cpp

  LINK_LIBS
    clangAST
    lldbBreakpoint
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessUtility
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage
    ${LLDB_CURSES_LIBS}

  LINK_COMPONENTS
    BinaryFormat
    Support
    Demangle
  )

# Needed to properly resolve references in a debug build.
# TODO: Remove once we have better layering
set_target_properties(lldbCore PROPERTIES LINK_INTERFACE_MULTIPLICITY 4)
