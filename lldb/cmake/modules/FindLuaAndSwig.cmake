#.rst:
# FindLuaAndSwig
# --------------
#
# Find Lua and SWIG as a whole.

if(LUA_LIBRARIES AND LUA_INCLUDE_DIR AND LLDB_ENABLE_SWIG)
  set(LUAANDSWIG_FOUND TRUE)
else()
  if (LLDB_ENABLE_SWIG)
    find_package(Lua 5.3)
    if(LUA_FOUND)
      # Find the Lua executable. Only required to run a subset of the Lua
      # tests.
      find_program(LUA_EXECUTABLE
        NAMES
        "lua"
        "lua${LUA_VERSION_MAJOR}.${LUA_VERSION_MINOR}"
      )
      mark_as_advanced(
        LUA_LIBRARIES
        LUA_INCLUDE_DIR
        LUA_VERSION_MINOR
        LUA_VERSION_MAJOR
        LUA_EXECUTABLE)
    endif()
  else()
    message(STATUS "SWIG 4 or later is required for Lua support in LLDB but could not be found")
  endif()


  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(LuaAndSwig
                                    FOUND_VAR
                                      LUAANDSWIG_FOUND
                                    REQUIRED_VARS
                                      LUA_LIBRARIES
                                      LUA_INCLUDE_DIR
                                      LUA_VERSION_MINOR
                                      LUA_VERSION_MAJOR
                                      LLDB_ENABLE_SWIG)
endif()
