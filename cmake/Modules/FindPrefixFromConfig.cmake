# Find the prefix from the `*Config.cmake` file being generated.
#
# When generating an installed `*Config.cmake` file, we often want to be able
# to refer to the ancestor directory which contains all the installed files.
#
# We want to do this without baking in an absolute path when the config file is
# generated, in order to allow for a "relocatable" binary distribution that
# doesn't need to know what path it ends up being installed at when it is
# built.
#
# The solution that we know the relative path that the config file will be at
# within that prefix, like `"${prefix_var}/lib/cmake/${project}"`, so we count
# the number of components in that path to figure out how many parent dirs we
# need to traverse from the location of the config file to get to the prefix
# dir.
#
# out_var:
#   variable to set the "return value" of the function, which is the code to
#   include in the config file under construction.
#
# prefix_var:
#   Name of the variable to define in the returned code (not directory for the
#   faller!) that will contain the prefix path.
#
# path_to_leave:
#   Path from the prefix to the config file, a relative path which we wish to
#   go up and out from to find the prefix directory.
function(find_prefix_from_config out_var prefix_var path_to_leave)
  if(IS_ABSOLUTE "${path_to_leave}")
    # Because the path is absolute, we don't care about `path_to_leave`
    # because we can just "jump" to the absolute path rather than work
    # our way there relatively.
    set(config_code
      "# Installation prefix is fixed absolute path"
      "set(${prefix_var} \"${CMAKE_INSTALL_PREFIX}\")")
  else()
    # `path_to_leave` is relative. Relative to what? The install prefix.
    # We therefore go up enough parent directories to get back to the
    # install prefix, and avoid hard-coding any absolute paths.
    set(config_code
      "# Compute the installation prefix from this LLVMConfig.cmake file location."
      "get_filename_component(${prefix_var} \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
    # Construct the proper number of get_filename_component(... PATH)
    # calls to compute the installation prefix.
    string(REGEX REPLACE "/" ";" _count "${path_to_leave}")
    foreach(p ${_count})
      list(APPEND config_code
        "get_filename_component(${prefix_var} \"\${${prefix_var}}\" PATH)")
    endforeach(p)
  endif()
  string(REPLACE ";" "\n" config_code "${config_code}")
  set("${out_var}" "${config_code}" PARENT_SCOPE)
endfunction()
