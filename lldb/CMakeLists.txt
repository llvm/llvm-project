cmake_minimum_required(VERSION 2.8)

include(cmake/modules/LLDBStandalone.cmake)
include(cmake/modules/LLDBConfig.cmake)
include(cmake/modules/AddLLDB.cmake)

#add_subdirectory(include)
add_subdirectory(docs)
if (NOT LLDB_DISABLE_PYTHON)
  add_subdirectory(scripts)
endif ()
add_subdirectory(source)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(unittests)
add_subdirectory(lit)

if ( LLDB_ENABLE_PYTHON_SCRIPTS_SWIG_API_GENERATION AND NOT LLDB_DISABLE_PYTHON )
	# Add a Post-Build Event to copy over Python files and create the symlink to liblldb.so for the Python API(hardlink on Windows)
    add_custom_target( finish_swig ALL
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/finishSwigWrapperClasses.py "--srcRoot=${LLDB_SOURCE_DIR}" "--targetDir=${CMAKE_CURRENT_BINARY_DIR}/scripts" "--cfgBldDir=${CMAKE_CURRENT_BINARY_DIR}/scripts" "--prefix=${CMAKE_BINARY_DIR}" "--cmakeBuildConfiguration=${CMAKE_CFG_INTDIR}" -m
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/finishSwigWrapperClasses.py
        COMMENT "Python script sym-linking LLDB Python API")
    add_dependencies(finish_swig liblldb argdumper)
endif ()
