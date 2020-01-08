#.rst:
# FindPythonInterpndLibs
# -----------
#
# Find the python interpreter and libraries as a whole.

if(PYTHON_LIBRARIES AND PYTHON_INCLUDE_DIRS AND PYTHON_EXECUTABLE)
  set(PYTHONINTERPANDLIBS_FOUND TRUE)
else()
  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if (Python3_FOUND AND Python3_Interpreter_FOUND)
      set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
      set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
      set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
      mark_as_advanced(
        PYTHON_LIBRARIES
        PYTHON_INCLUDE_DIRS
        PYTHON_EXECUTABLE)
    endif()
  else()
    find_package(PythonInterp QUIET)
    find_package(PythonLibs QUIET)
    if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
      if (NOT CMAKE_CROSSCOMPILING)
        string(REPLACE "." ";" pythonlibs_version_list ${PYTHONLIBS_VERSION_STRING})
        list(GET pythonlibs_version_list 0 pythonlibs_major)
        list(GET pythonlibs_version_list 1 pythonlibs_minor)

        # Ignore the patch version. Some versions of macOS report a different
        # patch version for the system provided interpreter and libraries.
        if (CMAKE_CROSSCOMPILING OR (PYTHON_VERSION_MAJOR VERSION_EQUAL pythonlibs_major AND
            PYTHON_VERSION_MINOR VERSION_EQUAL pythonlibs_minor))
          mark_as_advanced(
            PYTHON_LIBRARIES
            PYTHON_INCLUDE_DIRS
            PYTHON_EXECUTABLE)
        endif()
      endif()
    endif()
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(PythonInterpAndLibs
                                    FOUND_VAR
                                      PYTHONINTERPANDLIBS_FOUND
                                    REQUIRED_VARS
                                      PYTHON_LIBRARIES
                                      PYTHON_INCLUDE_DIRS
                                      PYTHON_EXECUTABLE)
endif()
