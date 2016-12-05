set(LLVM_NO_RTTI 1)

include(AddLLVM)
include(SwiftAddCustomCommandTarget)

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  add_definitions( -DEXPORT_LIBLLDB )
endif()

# Include this so that add_lldb_library() has the list of dependencies
# for liblldb to link against
include(${LLDB_PROJECT_ROOT}/cmake/LLDBDependencies.cmake)

option(LLDB_BUILD_FRAMEWORK "Build the Darwin LLDB.framework" Off)

if (LLDB_BUILD_FRAMEWORK AND NOT APPLE)
  message(FATAL_ERROR "LLDB.framework cannot be generated unless targeting Apple platforms.")
endif()

add_lldb_library(liblldb SHARED
  SBAddress.cpp
  SBAttachInfo.cpp
  SBBlock.cpp
  SBBreakpoint.cpp
  SBBreakpointLocation.cpp
  SBBroadcaster.cpp
  SBCommandInterpreter.cpp
  SBCommandReturnObject.cpp
  SBCommunication.cpp
  SBCompileUnit.cpp
  SBData.cpp
  SBDebugger.cpp
  SBDeclaration.cpp
  SBError.cpp
  SBEvent.cpp
  SBExecutionContext.cpp
  SBExpressionOptions.cpp
  SBFileSpec.cpp
  SBFileSpecList.cpp
  SBFrame.cpp
  SBFunction.cpp
  SBHostOS.cpp
  SBInstruction.cpp
  SBInstructionList.cpp
  SBLanguageRuntime.cpp
  SBLaunchInfo.cpp
  SBLineEntry.cpp
  SBListener.cpp
  SBMemoryRegionInfo.cpp
  SBMemoryRegionInfoList.cpp
  SBModule.cpp
  SBModuleSpec.cpp
  SBPlatform.cpp
  SBProcess.cpp
  SBQueue.cpp
  SBQueueItem.cpp
  SBSection.cpp
  SBSourceManager.cpp
  SBStream.cpp
  SBStringList.cpp
  SBStructuredData.cpp
  SBSymbol.cpp
  SBSymbolContext.cpp
  SBSymbolContextList.cpp
  SBTarget.cpp
  SBThread.cpp
  SBThreadCollection.cpp
  SBThreadPlan.cpp
  SBType.cpp
  SBTypeCategory.cpp
  SBTypeEnumMember.cpp
  SBTypeFilter.cpp
  SBTypeFormat.cpp
  SBTypeNameSpecifier.cpp
  SBTypeSummary.cpp
  SBTypeSynthetic.cpp
  SBValue.cpp
  SBValueList.cpp
  SBVariablesOptions.cpp
  SBWatchpoint.cpp
  SBUnixSignals.cpp
  SystemInitializerFull.cpp
  ${LLDB_WRAP_PYTHON}
  )

if (LLVM_ENABLE_WERROR)
  if (MSVC)
    set_property(SOURCE ${LLDB_WRAP_PYTHON} APPEND_STRING PROPERTY COMPILE_FLAGS " /W0")
  else()
    set_property(SOURCE ${LLDB_WRAP_PYTHON} APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
  endif()
endif()

# This should not be part of LLDBDependencies.cmake, because we don't
# want every single library taking a dependency on the script interpreters.
target_link_libraries(liblldb PRIVATE
  lldbPluginScriptInterpreterNone
  lldbPluginScriptInterpreterPython
  )

set_target_properties(liblldb
  PROPERTIES
  VERSION ${LLDB_VERSION}
  )

if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  if (NOT LLDB_EXPORT_ALL_SYMBOLS)
    # If we're not exporting all symbols, we'll want to explicitly set
    # the exported symbols here.  This prevents 'log enable --stack ...'
    # from working on some systems but limits the liblldb size.
    MESSAGE("-- Symbols (liblldb): only exporting symbols in lldb namespace")
    add_llvm_symbol_exports(liblldb ${CMAKE_CURRENT_SOURCE_DIR}/liblldb.exports)
  else()
    # Don't use an explicit export.  Instead, tell the linker to
    # export all symbols.
    MESSAGE("-- Symbols (liblldb): exporting symbols from lldb and lldb_private namespace")
    add_llvm_symbol_exports(liblldb ${CMAKE_CURRENT_SOURCE_DIR}/liblldb-private.exports)
  endif()
endif()

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  # Only MSVC has the ABI compatibility problem and avoids using FindPythonLibs,
  # so only it needs to explicitly link against ${PYTHON_LIBRARY}
  if (MSVC AND NOT LLDB_DISABLE_PYTHON)
    target_link_libraries(liblldb PRIVATE ${PYTHON_LIBRARY})
  endif()
else()
  set_target_properties(liblldb
    PROPERTIES
    OUTPUT_NAME lldb
    )
endif()

if (LLDB_WRAP_PYTHON)
  add_dependencies(liblldb swig_wrapper)
endif()
target_link_libraries(liblldb PRIVATE ${LLDB_SYSTEM_LIBS})

set(lib_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX}")
set(CLANG_RESOURCE_PATH "${LLDB_PATH_TO_SWIFT_BUILD}/lib${LLVM_LIBDIR_SUFFIX}/swift/clang")
set(clang_headers_target ${CLANG_RESOURCE_PATH}/include)
if(NOT LLDB_BUILT_STANDALONE)
  set(clang_headers_target symlink_clang_headers)
endif()

add_custom_command_target(
    unused_var
    COMMAND "${CMAKE_COMMAND}" "-E" "copy_directory" "${CLANG_RESOURCE_PATH}" "${lib_dir}/lldb/clang"
    OUTPUT "${lib_dir}/lldb/clang"
    VERBATIM
    ALL
    DEPENDS ${clang_headers_target})

if(LLDB_BUILD_FRAMEWORK)
  file(GLOB public_headers ${LLDB_SOURCE_DIR}/include/lldb/API/*.h)
  set_target_properties(liblldb PROPERTIES
    OUTPUT_NAME LLDB
    FRAMEWORK On
    FRAMEWORK_VERSION ${LLDB_FRAMEWORK_VERSION}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}
    PUBLIC_HEADER "${public_headers}")

  set(symlink_commands
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:liblldb>
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${LLDB_SOURCE_DIR}/include/lldb/API $<TARGET_FILE_DIR:liblldb>/Headers
    COMMAND ${CMAKE_COMMAND} -E create_symlink Versions/Current/Headers ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Headers 
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/clang/${LLDB_VERSION} $<TARGET_FILE_DIR:liblldb>/Resources/Clang
    )

  # This works around a CMake bug where POST_BUILD steps are not applied to 
  # framework targets. This fix is merged into the CMake release branch and
  # should be available with CMake 3.7 rc2:
  # https://gitlab.kitware.com/cmake/cmake/issues/16363
  if(CMAKE_VERSION VERSION_GREATER 3.6.99)
    add_custom_command(TARGET liblldb POST_BUILD ${symlink_commands})
  else()
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Versions/${LLDB_FRAMEWORK_VERSION}/Headers
      ${symlink_commands})
    add_custom_target(lldb_header_symlink
      DEPENDS ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Versions/${LLDB_FRAMEWORK_VERSION}/Headers)
  endif()
endif()

install(
  CODE "file(MAKE_DIRECTORY ${lib_dir}/lldb)")

install(
  DIRECTORY "${lib_dir}/lldb/clang"
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/lldb/)
