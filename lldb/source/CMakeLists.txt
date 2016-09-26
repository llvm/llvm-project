include_directories(.)

if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
include_directories(
  Plugins/Process/Linux
  Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
include_directories(
  Plugins/Process/FreeBSD
  Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "NetBSD" )
include_directories(
  Plugins/Process/POSIX
  )
endif ()


set(lldbBase_SOURCES
    lldb.cpp
  )

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(LLDB_VERS_GENERATED_FILE ${LLDB_BINARY_DIR}/source/LLDB_vers.c)
  add_custom_command(OUTPUT ${LLDB_VERS_GENERATED_FILE}
    COMMAND ${LLDB_SOURCE_DIR}/scripts/generate-vers.pl
            ${LLDB_SOURCE_DIR}/lldb.xcodeproj/project.pbxproj liblldb_core
            > ${LLDB_VERS_GENERATED_FILE}
    DEPENDS ${LLDB_SOURCE_DIR}/scripts/generate-vers.pl
            ${LLDB_SOURCE_DIR}/lldb.xcodeproj/project.pbxproj)
  set_source_files_properties(${LLDB_VERS_GENERATED_FILE} PROPERTIES GENERATED 1)
  # Add this to lldbBase since lldb.cpp uses the symbol defined here.
  list(APPEND lldbBase_SOURCES ${LLDB_VERS_GENERATED_FILE})
  add_custom_target(lldbGeneratedVersion
    DEPENDS ${LLDB_VERS_GENERATED_FILE})
endif()

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
  set(get_svn_script "${LLVM_MAIN_SRC_DIR}/cmake/modules/GetSVN.cmake")

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
  set_source_files_properties(lldb.cpp
    PROPERTIES COMPILE_DEFINITIONS "HAVE_SVN_VERSION_INC")
  list(APPEND lldbBase_SOURCES ${version_inc})
endif()

add_lldb_library(lldbBase
  ${lldbBase_SOURCES}
  )

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_dependencies(lldbBase lldbGeneratedVersion)
endif()

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
