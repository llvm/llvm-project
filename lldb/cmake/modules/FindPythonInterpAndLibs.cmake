#.rst:
# FindPythonInterpAndLibs
# -----------
#
# Find the python interpreter and libraries as a whole.

if(PYTHON_LIBRARIES AND PYTHON_INCLUDE_DIRS AND PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
  set(PYTHONINTERPANDLIBS_FOUND TRUE)
else()
  find_package(SWIG 2.0)
  if (SWIG_FOUND)
    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
      # Use PYTHON_HOME as a hint to find Python 3.
      set(Python3_ROOT_DIR PYTHON_HOME)
      find_package(Python3 COMPONENTS Interpreter Development)
      if (Python3_FOUND AND Python3_Interpreter_FOUND)
        set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
        set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
        set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
        mark_as_advanced(
          PYTHON_LIBRARIES
          PYTHON_INCLUDE_DIRS
          PYTHON_EXECUTABLE
          SWIG_EXECUTABLE)
      endif()
    else()
      find_package(PythonInterp)
      find_package(PythonLibs)
      if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND SWIG_FOUND)
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
              PYTHON_EXECUTABLE
              SWIG_EXECUTABLE)
          endif()
        endif()
      endif()
    endif()
  else()
    message(STATUS "SWIG 2 or later is required for Python support in LLDB but could not be found")
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(PythonInterpAndLibs
                                    FOUND_VAR
                                      PYTHONINTERPANDLIBS_FOUND
                                    REQUIRED_VARS
                                      PYTHON_LIBRARIES
                                      PYTHON_INCLUDE_DIRS
                                      PYTHON_EXECUTABLE
                                      SWIG_EXECUTABLE)
endif()
