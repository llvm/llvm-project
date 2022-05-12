lldb_tablegen(CoreProperties.inc -gen-lldb-property-defs
  SOURCE CoreProperties.td
  TARGET LLDBCorePropertiesGen)

lldb_tablegen(CorePropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE CoreProperties.td
  TARGET LLDBCorePropertiesEnumGen)

set(LLDB_CURSES_LIBS)
set(LLDB_LIBEDIT_LIBS)

if (LLDB_ENABLE_CURSES)
  list(APPEND LLDB_CURSES_LIBS ${CURSES_LIBRARIES} ${PANEL_LIBRARIES})
  if(LLVM_ENABLE_TERMINFO)
    list(APPEND LLDB_CURSES_LIBS ${TERMINFO_LIB})
  endif()
  if (LLVM_BUILD_STATIC)
    list(APPEND LLDB_CURSES_LIBS gpm)
  endif()
endif()

add_lldb_library(lldbCore
  Address.cpp
  AddressRange.cpp
  AddressResolver.cpp
  AddressResolverFileLine.cpp
  Communication.cpp
  DataFileCache.cpp
  Debugger.cpp
  DebuggerEvents.cpp
  Declaration.cpp
  Disassembler.cpp
  DumpDataExtractor.cpp
  DumpRegisterValue.cpp
  DynamicLoader.cpp
  EmulateInstruction.cpp
  FileLineResolver.cpp
  FileSpecList.cpp
  FormatEntity.cpp
  Highlighter.cpp
  IOHandler.cpp
  IOHandlerCursesGUI.cpp
  Mangled.cpp
  Module.cpp
  ModuleChild.cpp
  ModuleList.cpp
  Opcode.cpp
  PluginManager.cpp
  Progress.cpp
  RichManglingContext.cpp
  SearchFilter.cpp
  Section.cpp
  SourceLocationSpec.cpp
  SourceManager.cpp
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
  ValueObjectUpdater.cpp
  ValueObjectVariable.cpp

  DEPENDS
    clang-tablegen-targets

  LINK_LIBS
    lldbBreakpoint
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage
    ${LLDB_CURSES_LIBS}

  CLANG_LIBS
    clangDriver

  LINK_COMPONENTS
    Support
    Demangle
  )

add_dependencies(lldbCore
  LLDBCorePropertiesGen
  LLDBCorePropertiesEnumGen)

# Needed to properly resolve references in a debug build.
# TODO: Remove once we have better layering
set_target_properties(lldbCore PROPERTIES LINK_INTERFACE_MULTIPLICITY 5)

if (LLDB_ENABLE_CURSES)
  target_include_directories(lldbCore PRIVATE ${CURSES_INCLUDE_DIRS})
endif()
