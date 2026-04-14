lldb_tablegen(CoreProperties.inc -gen-lldb-property-defs
  SOURCE CoreProperties.td
  TARGET LLDBCorePropertiesGen)

lldb_tablegen(CorePropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE CoreProperties.td
  TARGET LLDBCorePropertiesEnumGen)

lldb_tablegen(CoreProperties.json -dump-json
  SOURCE CoreProperties.td
  TARGET LLDBCorePropertiesJsonGen)

set(LLDB_CURSES_LIBS)
set(LLDB_LIBEDIT_LIBS)

if (LLDB_ENABLE_CURSES)
  list(APPEND LLDB_CURSES_LIBS ${PANEL_LIBRARIES} ${CURSES_LIBRARIES})
  if (LLVM_BUILD_STATIC)
    list(APPEND LLDB_CURSES_LIBS gpm)
  endif()
endif()

add_lldb_library(lldbCore NO_PLUGIN_DEPENDENCIES
  Address.cpp
  AddressRange.cpp
  AddressRangeListImpl.cpp
  AddressResolver.cpp
  AddressResolverFileLine.cpp
  Communication.cpp
  DataFileCache.cpp
  Debugger.cpp
  DebuggerEvents.cpp
  Declaration.cpp
  DemangledNameInfo.cpp
  Disassembler.cpp
  DumpDataExtractor.cpp
  DumpRegisterValue.cpp
  DumpRegisterInfo.cpp
  DynamicLoader.cpp
  EmulateInstruction.cpp
  FileLineResolver.cpp
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
  ProtocolServer.cpp
  Statusline.cpp
  RichManglingContext.cpp
  SearchFilter.cpp
  Section.cpp
  SourceLocationSpec.cpp
  SourceManager.cpp
  StreamAsynchronousIO.cpp
  Telemetry.cpp
  ThreadedCommunication.cpp
  UserSettingsController.cpp
  Value.cpp

  ADDITIONAL_HEADER_DIRS
    ${LLDB_INCLUDE_DIR}/lldb/Core
  DEPENDS
    clang-tablegen-targets
  LINK_COMPONENTS
    Support
    Demangle
    TargetParser
    Telemetry
  LINK_LIBS
    lldbBreakpoint
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbValueObject
    lldbVersion
    ${LLDB_CURSES_LIBS}

  CLANG_LIBS
    clangDriver
  )

set(LLDB_SAFE_AUTO_LOAD_PATHS "" CACHE STRING
    "Semicolon-separated list of paths that LLDB will automatically load scripting resources from.")

# Turn list of paths into a comma-separated list to be #include'd
# into a C-array.
set(_entries "")
foreach(path IN LISTS LLDB_SAFE_AUTO_LOAD_PATHS)
  string(REPLACE "\\" "\\\\" escaped_path "${path}")
  list(APPEND _entries "\"${escaped_path}\",")
endforeach()
string(JOIN "\n" SAFE_PATH_ENTRIES ${_entries})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SafeAutoloadPaths.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/SafeAutoloadPaths.inc
  @ONLY
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
