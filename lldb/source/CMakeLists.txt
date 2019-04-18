include_directories(.)

set(lldbBase_SOURCES
    lldb.cpp
  )

foreach(file
        "${LLDB_SOURCE_DIR}/.git/logs/HEAD" # Git
        "${LLDB_SOURCE_DIR}/.svn/wc.db"     # SVN 1.7
        "${LLDB_SOURCE_DIR}/.svn/entries"   # SVN 1.6
        )
  if(EXISTS "${file}")
    set(lldb_vc "${file}")
    break()
  endif()
endforeach()

if(DEFINED lldb_vc)
  set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc")
  set(get_svn_script "${LLVM_CMAKE_PATH}/GetSVN.cmake")

  # Create custom target to generate the VC revision include.
  add_custom_command(OUTPUT "${version_inc}"
    DEPENDS "${lldb_vc}" "${get_svn_script}"
    COMMAND
    ${CMAKE_COMMAND} "-DFIRST_SOURCE_DIR=${LLDB_SOURCE_DIR}"
                     "-DFIRST_NAME=LLDB"
                     "-DHEADER_FILE=${version_inc}"
                     -P "${get_svn_script}")

  # Mark the generated header as being generated.
  set_source_files_properties("${version_inc}"
    PROPERTIES GENERATED TRUE
               HEADER_FILE_ONLY TRUE)

  # Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
  set_property(SOURCE lldb.cpp APPEND PROPERTY 
               COMPILE_DEFINITIONS "HAVE_SVN_VERSION_INC")
  list(APPEND lldbBase_SOURCES ${version_inc})
endif()

if(LLDB_VERSION_STRING)
  set_property(SOURCE lldb.cpp APPEND PROPERTY
               COMPILE_DEFINITIONS "LLDB_VERSION_STRING=${LLDB_VERSION_STRING}")
endif()

add_lldb_library(lldbBase
  ${lldbBase_SOURCES}
  )

add_subdirectory(Breakpoint)
add_subdirectory(Commands)
add_subdirectory(Core)
add_subdirectory(DataFormatters)
add_subdirectory(Expression)
add_subdirectory(Host)
add_subdirectory(Initialization)
add_subdirectory(Interpreter)
add_subdirectory(Plugins)
add_subdirectory(Symbol)
add_subdirectory(Target)
add_subdirectory(Utility)

# Build API last.  Since liblldb needs to link against every other target, it needs
# those targets to have already been created.
add_subdirectory(API)
