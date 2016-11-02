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

# Keep track of whether this is a buildbot build.
set(LLDB_IS_BUILDBOT_BUILD FALSE CACHE BOOL
  "Indicate whether this is a buildbot build")

# Keep track of when we did this build.  Note incremental
# builds could get burnt by this.
string(TIMESTAMP INTERNAL_BUILD_DATE "%Y-%m-%d")

set(lldbBase_SOURCES
    lldb.cpp
  )

set(LLDB_CPP_COMPILE_DEFINITIONS
  "LLDB_IS_BUILDBOT_BUILD=${LLDB_IS_BUILDBOT_BUILD}"
  "LLDB_BUILD_DATE=\"${INTERNAL_BUILD_DATE}\""
  "LLDB_USE_OSS_VERSION_SCHEME"
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

# Determine LLDB revision and repository. GetSourceVersion and GetRepositoryPath are shell-scripts, and as
# such will not work on Windows.  The revision piece is now Python, so
# that works (git/svn).
if (PYTHON_EXECUTABLE)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${LLDB_SOURCE_DIR}/scripts/get_vcs_revision.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE LLDB_REVISION)
  if ( LLDB_REVISION )
    string(REGEX REPLACE "(\r?\n)+$" "" LLDB_REVISION ${LLDB_REVISION})
    set(LLDB_CPP_COMPILE_DEFINITIONS
      "${LLDB_CPP_COMPILE_DEFINITIONS}"
      "LLDB_REVISION=\"${LLDB_REVISION}\""
      )
  endif()
endif()

if ( NOT CMAKE_SYSTEM_NAME MATCHES "Windows" )
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/utils/GetRepositoryPath ${LLDB_SOURCE_DIR}
                  OUTPUT_VARIABLE LLDB_REPOSITORY)
  if ( LLDB_REPOSITORY )
    # Replace newline characters with spaces
    string(REGEX REPLACE "(\r?\n)+" " " LLDB_REPOSITORY ${LLDB_REPOSITORY})

    # Remove trailing spaces
    string(REGEX REPLACE "(\ )+$" "" LLDB_REPOSITORY ${LLDB_REPOSITORY})

    set(LLDB_CPP_COMPILE_DEFINITIONS
      "${LLDB_CPP_COMPILE_DEFINITIONS}"
      "LLDB_REPOSITORY=\"${LLDB_REPOSITORY}\""
      )
  endif()
endif()

set_property(
  SOURCE lldb.cpp
  PROPERTY COMPILE_DEFINITIONS ${LLDB_CPP_COMPILE_DEFINITIONS})
# FIXME: implement svn/git revision and repository parsing solution on Windows. There is an SVN-only
#        revision parsing solution in tools/clang/lib/Basic/CMakelists.txt.
