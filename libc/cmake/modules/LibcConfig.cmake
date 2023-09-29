# This cmake module contains utilities to read and load libc config options
# listed in config.json files.
#
# The JSON parsing commands that CMake provides are rather tedious to use.
# Below is a quick reference which tries to map the CMake JSON parsing
# commands to the Python dictionary API.
#
# * There is no way to iterate over the JSON items. One will first
#   have to find the number of items using string(JSON ... LENGTH ...)
#   command, and then iterate over the items using foreach(... RANGE ...).
# * The way to get the key from the JSON dictionary is to use the index
#   of the item and the string(JSON ... MEMBER ... $<index>) function.
# * Once you have the key, you can use the string(JSON ... GET ... $<key>)
#   function to get the value corresponding to the key.

# Fill |opt_list| with all options listed in |config_file|. For each option,
# the item added to |opt_list| is the dictionary of the form:
#   {
#     "<option name>": {
#       "value: <option value>,
#       "doc": "<option doc string>",
#     }
#   }
# Each of the above items can be parsed again with the string(JSON ...)
# command.
# This function does nothing if |config_file| is missing.
function(read_libc_config config_file opt_list)
  if(NOT EXISTS ${config_file})
    return()
  endif()
  # We will assume that a config file is loaded only once and that
  # each config file loaded will affect config information. Since
  # we want a change to config information to trigger reconfiguration,
  # we add the |config_file| to the list of files the configure itself
  # should depend on.
  set_property(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${config_file})

  file(READ ${config_file} json_config)
  string(JSON group_count ERROR_VARIABLE json_error LENGTH ${json_config})
  if(json_error)
    message(FATAL_ERROR "${config_file}: ${json_error}")
  endif()
  if(${group_count} EQUAL 0)
    # This "if" conditions becomes active if there are no config options
    # to load. If there are no config options, it is better to remove that
    # config.json file instead of including an empty file.
    message(FATAL_ERROR "${config_file}: Does not contain any config option groups")
  endif()
  math(EXPR group_count_1 "${group_count} - 1")

  set(optname_list)
  foreach(group_num RANGE ${group_count_1})
    # The group names are the keys of the global dictionary. So, we first
    # lookup the group name or the key for each item in the dictionary.
    string(JSON group_name ERROR_VARIABLE json_error MEMBER ${json_config} ${group_num})
    if(json_error)
      message(FATAL_ERROR "${config_file}: ${json_error}")
    endif()

    # Once we have the group name, we GET the option map for that group, which
    # is the value corresponding to the group name key.
    string(JSON option_map ERROR_VARIABLE json_error GET ${json_config} ${group_name})
    if(json_error)
      message(FATAL_ERROR ${json_error})
    endif()
    string(JSON option_count ERROR_VARIABLE jsor_error LENGTH ${option_map})
    if(json_error)
      message(FATAL_ERROR ${json_error})
    endif()
    if(${option_count} EQUAL 0)
      message(FATAL_ERROR "${config_file}: No options listed against the config option group '${group_name}'")
    endif()

    math(EXPR option_count_1 "${option_count} - 1")
    foreach(opt_num RANGE ${option_count_1})
      string(JSON option_name ERROR_VARIABLE json_error MEMBER ${option_map} ${opt_num})
      if(json_error)
        message(FATAL_ERROR ${json_error})
      endif()
      list(FIND optname_list ${option_name} optname_exists)
      if(${optname_exists} GREATER -1)
        message(FATAL_ERROR "${config_file}: Found duplicate option name: ${option_name}")
      endif()
      list(APPEND optname_list ${option_name})

      string(JSON optdata ERROR_VARIABLE json_error GET ${option_map} ${option_name})
      if(json_error)
        message(FATAL_ERROR ${json_error})
      endif()
      set(opt "{\"${option_name}\": ${optdata}}")
      list(APPEND all_opts ${opt})
    endforeach()
  endforeach()
  set(${opt_list} ${all_opts} PARENT_SCOPE)
endfunction()

# Loads the config options listed in |config_file| in the following way:
# * For each option listed in the |config_file|, it looks for existence of a
#   var with the same name. It is an error if the var is not already defined.
#   If a var with the option name is found, then its value is overwritten
#   with the value specified in |config_file|.
# * If there are options which are not to be overriden, then the list of
#   such options can be passed to this function after the |config_file|
#   argument. Typically, these will be the options specified on the CMake
#   command line.
function(load_libc_config config_file)
  read_libc_config(${config_file} file_opts)
  foreach(opt IN LISTS file_opts)
    string(JSON opt_name ERROR_VARIABLE json_error MEMBER ${opt} 0)
    if(json_error)
      message(FATAL_ERROR ${json_error})
    endif()
    if(NOT DEFINED ${opt_name})
      message(FATAL_ERROR: " Option ${opt_name} defined in ${config_file} is invalid.")
    endif()
    if(ARGN)
      list(FIND ARGN ${opt_name} optname_exists)
      if(${optname_exists} GREATER -1)
        # This option is not to be overridden so just skip further processing.
        continue()
      endif()
    endif()
    string(JSON opt_object ERROR_VARIABLE json_error GET ${opt} ${opt_name})
    if(json_error)
      message(FATAL_ERROR ${json_error})
    endif()
    string(JSON opt_value ERROR_VARIABLE jsor_error GET ${opt_object} "value")
    if(json_error)
      message(FATAL_ERROR ${json_error})
    endif()
    message(STATUS "Overriding - ${opt_name}: ${opt_value} (Previous value: ${${opt_name}})")
    set(${opt_name} ${opt_value} PARENT_SCOPE)
  endforeach()
endfunction()

function(generate_config_doc config_file doc_file)
  if(NOT EXISTS ${config_file})
    message(FATAL_ERROR "${config_file} does not exist")
  endif()
  file(READ ${config_file} json_config)
  string(JSON group_count ERROR_VARIABLE json_error LENGTH ${json_config})
  if(json_error)
    message(FATAL_ERROR "${config_file}: ${json_error}")
  endif()
  if(${group_count} EQUAL 0)
    message(FATAL_ERROR "${config_file}: Does not contain any config option groups")
  endif()
  math(EXPR group_count_1 "${group_count} - 1")

  set(doc_string ".. _configure:\n"
                 "..\n"
                 "   Do not edit this file directly. CMake will auto generate it.\n"
                 "   If the changes are intended, add this file to your commit.\n"
                 "\n"
                 "==========================\n"
                 "Configure Options\n"
                 "==========================\n"
                 "\n"
                 "Below is the full set of options one can use to configure the libc build.\n"
                 "An option can be given an explicit value on the CMake command line using\n"
                 "the following syntax:\n"
                 "\n"
                 ".. code-block:: sh\n"
                 "\n"
                 "  $> cmake <other build options> -D<libc config option name>=<option value> <more options>\n"
                 "\n"
                 "For example:\n"
                 "\n"
                 ".. code-block:: sh\n"
                 "\n"
                 "  $> cmake <other build options> -DLIBC_CONF_PRINTF_DISABLE_FLOAT=ON <more options>\n"
                 "\n"
                 "See the main ``config/config.json``, and the platform and architecture specific\n"
                 "overrides in ``config/<platform>/config.json`` and ``config/<platform>/<arch>/config.json,``\n"
                 "to learn about the defaults for your platform and target.\n"
                 "\n")

  foreach(group_num RANGE ${group_count_1})
    string(JSON group_name ERROR_VARIABLE json_error MEMBER ${json_config} ${group_num})
    if(json_error)
      message(FATAL_ERROR "${config_file}: ${json_error}")
    endif()
    string(APPEND doc_string "* **\"${group_name}\" options**\n")
    string(JSON option_map ERROR_VARIABLE json_error GET ${json_config} ${group_name})
    if(json_error)
      message(FATAL_ERROR ${json_error})
    endif()
    string(JSON option_count ERROR_VARIABLE jsor_error LENGTH ${option_map})
    if(json_error)
      message(FATAL_ERROR ${json_error})
    endif()
    if(${option_count} EQUAL 0)
      message(FATAL_ERROR "${config_file}: No options listed against the config option group '${group_name}'")
    endif()

    math(EXPR option_count_1 "${option_count} - 1")
    foreach(opt_num RANGE ${option_count_1})
      string(JSON option_name ERROR_VARIABLE json_error MEMBER ${option_map} ${opt_num})
      if(json_error)
        message(FATAL_ERROR ${json_error})
      endif()
      string(JSON opt_object ERROR_VARIABLE json_error GET ${option_map} ${option_name})
      if(json_error)
        message(FATAL_ERROR "Error generating ${doc_file}: ${json_error}\n${opt_object}")
      endif()
      string(JSON opt_doc ERROR_VARIABLE json_error GET ${opt_object} "doc")
      if(json_error)
        message(FATAL_ERROR "Error generating ${doc_file}: ${json_error}")
      endif()
      string(APPEND doc_string "    - ``${option_name}``: ${opt_doc}\n")
    endforeach()
  endforeach()
  message(STATUS "Writing config doc to ${doc_file}")
  file(WRITE ${doc_file} ${doc_string})
endfunction()
