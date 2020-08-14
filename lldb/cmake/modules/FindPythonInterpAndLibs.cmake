#.rst:
# FindPythonInterpAndLibs
# -----------
#
# Find the python interpreter and libraries as a whole.

macro(FindPython3)
  # Use PYTHON_HOME as a hint to find Python 3.
  set(Python3_ROOT_DIR "${PYTHON_HOME}")
  find_package(Python3 COMPONENTS Interpreter Development)
  if(Python3_FOUND AND Python3_Interpreter_FOUND)
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

    # The install name for the Python 3 framework in Xcode is relative to
    # the framework's location and not the dylib itself.
    #
    #   @rpath/Python3.framework/Versions/3.x/Python3
    #
    # This means that we need to compute the path to the Python3.framework
    # and use that as the RPATH instead of the usual dylib's directory.
    #
    # The check below shouldn't match Homebrew's Python framework as it is
    # called Python.framework instead of Python3.framework.
    if (APPLE AND Python3_LIBRARIES MATCHES "Python3.framework")
      string(FIND "${Python3_LIBRARIES}" "Python3.framework" python_framework_pos)
      string(SUBSTRING "${Python3_LIBRARIES}" "0" ${python_framework_pos} PYTHON_RPATH)
    endif()

    set(PYTHON3_FOUND TRUE)
    mark_as_advanced(
      PYTHON_LIBRARIES
      PYTHON_INCLUDE_DIRS
      PYTHON_EXECUTABLE
      PYTHON_RPATH
      SWIG_EXECUTABLE)
  endif()
endmacro()

if(PYTHON_LIBRARIES AND PYTHON_INCLUDE_DIRS AND PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
  set(PYTHONINTERPANDLIBS_FOUND TRUE)
else()
  find_package(SWIG 2.0)
  if (SWIG_FOUND)
      FindPython3()
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
