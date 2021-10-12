#.rst:
# FindLuaAndSwig
# --------------
#
# Find Lua and SWIG as a whole.

if(LUA_LIBRARIES AND LUA_INCLUDE_DIR AND SWIG_EXECUTABLE)
  set(LUAANDSWIG_FOUND TRUE)
else()
  find_package(SWIG 3.0 QUIET)
  if (SWIG_FOUND OR LLDB_USE_STATIC_BINDINGS)
    if (LLDB_USE_STATIC_BINDINGS)
      set(SWIG_EXECUTABLE "/not/found")
    endif()
    find_package(Lua 5.3 EXACT)
    if(LUA_FOUND AND SWIG_FOUND)
      mark_as_advanced(
        LUA_LIBRARIES
        LUA_INCLUDE_DIR
        SWIG_EXECUTABLE)
    endif()
  else()
    message(STATUS "SWIG 3 or later is required for Lua support in LLDB but could not be found")
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(LuaAndSwig
                                    FOUND_VAR
                                      LUAANDSWIG_FOUND
                                    REQUIRED_VARS
                                      LUA_LIBRARIES
                                      LUA_INCLUDE_DIR
                                      SWIG_EXECUTABLE)
endif()
