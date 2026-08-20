include(GetSubprojectTitle)


# This function canonicalize the CMake variables passed by names
# from CMake boolean to 0/1 suitable for passing into Python or C++,
# in place.
function(llvm_canonicalize_cmake_booleans)
  foreach(var ${ARGN})
    if(${var})
      set(${var} 1 PARENT_SCOPE)
    else()
      set(${var} 0 PARENT_SCOPE)
    endif()
  endforeach()
endfunction(llvm_canonicalize_cmake_booleans)

macro(set_llvm_build_mode)
  # Configuration-time: See Unit/lit.site.cfg.in
  if (CMAKE_CFG_INTDIR STREQUAL ".")
    set(LLVM_BUILD_MODE ".")
  else ()
    set(LLVM_BUILD_MODE "%(build_mode)s")
  endif ()
endmacro()

# Takes a list of path names in pathlist and a base directory, and returns
# a list of paths relative to the base directory in out_pathlist.
# Paths that are on a different drive than the basedir (on Windows) or that
# contain symlinks are returned absolute.
# Use with LLVM_LIT_PATH_FUNCTION below.
function(make_paths_relative out_pathlist basedir pathlist)
  # Passing ARG_PATH_VALUES as-is to execute_process() makes cmake strip
  # empty list entries. So escape the ;s in the list and do the splitting
  # ourselves. cmake has no relpath function, so use Python for that.
  string(REPLACE ";" "\\;" pathlist_escaped "${pathlist}")
  execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "\n
import os, sys\n
base = sys.argv[1]
def haslink(p):\n
    if not p or p == os.path.dirname(p): return False\n
    return os.path.islink(p) or haslink(os.path.dirname(p))\n
def relpath(p):\n
    if not p: return ''\n
    if os.path.splitdrive(p)[0] != os.path.splitdrive(base)[0]: return p\n
    if haslink(p) or haslink(base): return p\n
    return os.path.relpath(p, base)\n
if len(sys.argv) < 3: sys.exit(0)\n
sys.stdout.write(';'.join(relpath(p) for p in sys.argv[2].split(';')))"
    ${basedir}
    ${pathlist_escaped}
    OUTPUT_VARIABLE pathlist_relative
    ERROR_VARIABLE error
    RESULT_VARIABLE result)
  if (NOT result EQUAL 0)
    message(FATAL_ERROR "make_paths_relative() failed due to error '${result}', with stderr\n${error}")
  endif()
  set(${out_pathlist} "${pathlist_relative}" PARENT_SCOPE)
endfunction()

# Converts a file that's relative to the current python file to an absolute
# path. Since this uses __file__, it has to be emitted into python files that
# use it and can't be in a lit module. Use with make_paths_relative().
string(CONCAT LLVM_LIT_PATH_FUNCTION
  "# Allow generated file to be relocatable.\n"
  "import os\n"
  "import platform\n"
  "def path(p):\n"
  "    if not p: return ''\n"
  "    # Follows lit.util.abs_path_preserve_drive, which cannot be imported here.\n"
  "    if platform.system() == 'Windows':\n"
  "        return os.path.abspath(os.path.join(os.path.dirname(__file__), p))\n"
  "    else:\n"
  "        return os.path.realpath(os.path.join(os.path.dirname(__file__), p))\n"
  )

# This function provides an automatic way to 'configure'-like generate a file
# based on a set of common and custom variables, specifically targeting the
# variables needed for the 'lit.site.cfg' files. This function bundles the
# common variables that any Lit instance is likely to need, and custom
# variables can be passed in.
# The keyword PATHS is followed by a list of cmake variable names that are
# mentioned as `path("@varname@")` in the lit.cfg.py.in file. Variables in that
# list are treated as paths that are relative to the directory the generated
# lit.cfg.py file is in, and the `path()` function converts the relative
# path back to absolute form. This makes it possible to move a build directory
# containing lit.cfg.py files from one machine to another.
function(configure_lit_site_cfg site_in site_out)
  cmake_parse_arguments(ARG "" "" "MAIN_CONFIG;PATHS" ${ARGN})

  if ("${ARG_MAIN_CONFIG}" STREQUAL "")
    get_filename_component(INPUT_DIR ${site_in} DIRECTORY)
    set(ARG_MAIN_CONFIG "${INPUT_DIR}/lit.cfg")
  endif()

  foreach(c ${LLVM_TARGETS_TO_BUILD})
    set(TARGETS_BUILT "${TARGETS_BUILT} ${c}")
  endforeach(c)
  set(TARGETS_TO_BUILD ${TARGETS_BUILT})

  set(SHLIBEXT "${LTDL_SHLIB_EXT}")

  set_llvm_build_mode()

  # For standalone builds of subprojects, these might not be the build tree but
  # a provided binary tree.
  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(LLVM_BINARY_DIR ${LLVM_BINARY_DIR})
  string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
  string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" LLVM_LIBS_DIR  "${LLVM_LIBRARY_DIR}")
  # Like LLVM_{TOOLS,LIBS}_DIR, but pointing at the build tree.
  string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" CURRENT_TOOLS_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")
  string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" CURRENT_LIBS_DIR  "${LLVM_LIBRARY_OUTPUT_INTDIR}")
  string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" SHLIBDIR "${LLVM_SHLIB_OUTPUT_INTDIR}")

  # FIXME: "ENABLE_SHARED" doesn't make sense, since it is used just for
  # plugins. We may rename it.
  if(LLVM_ENABLE_PLUGINS)
    set(ENABLE_SHARED "1")
  else()
    set(ENABLE_SHARED "0")
  endif()

  if(LLVM_ENABLE_ASSERTIONS)
    set(ENABLE_ASSERTIONS "1")
  else()
    set(ENABLE_ASSERTIONS "0")
  endif()

  set(HOST_OS ${CMAKE_SYSTEM_NAME})
  set(HOST_ARCH ${CMAKE_SYSTEM_PROCESSOR})

  set(HOST_CC "${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}")
  set(HOST_CXX "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
  set(HOST_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

  string(CONCAT LIT_SITE_CFG_IN_HEADER
    "# Autogenerated from ${site_in}\n# Do not edit!\n\n"
    "${LLVM_LIT_PATH_FUNCTION}"
    )

  # Override config_target_triple (and the env)
  if(LLVM_TARGET_TRIPLE_ENV)
    # This is expanded into the heading.
    string(CONCAT LIT_SITE_CFG_IN_HEADER "${LIT_SITE_CFG_IN_HEADER}"
      "import os\n"
      "target_env = \"${LLVM_TARGET_TRIPLE_ENV}\"\n"
      "config.target_triple = config.environment[target_env] = os.environ.get(target_env, \"${LLVM_TARGET_TRIPLE}\")\n"
      )

    # This is expanded to; config.target_triple = ""+config.target_triple+""
    set(LLVM_TARGET_TRIPLE "\"+config.target_triple+\"")
  endif()

  if (ARG_PATHS)
    # Walk ARG_PATHS and collect the current value of the variables in there.
    # list(APPEND) ignores empty elements exactly if the list is empty,
    # so start the list with a dummy element and drop it, to make sure that
    # even empty values make it into the values list.
    set(ARG_PATH_VALUES "dummy")
    foreach(path ${ARG_PATHS})
      list(APPEND ARG_PATH_VALUES "${${path}}")
    endforeach()
    list(REMOVE_AT ARG_PATH_VALUES 0)

    get_filename_component(OUTPUT_DIR ${site_out} DIRECTORY)
    make_paths_relative(
        ARG_PATH_VALUES_RELATIVE "${OUTPUT_DIR}" "${ARG_PATH_VALUES}")

    list(LENGTH ARG_PATHS len_paths)
    list(LENGTH ARG_PATH_VALUES len_path_values)
    list(LENGTH ARG_PATH_VALUES_RELATIVE len_path_value_rels)
    if ((NOT ${len_paths} EQUAL ${len_path_values}) OR
        (NOT ${len_paths} EQUAL ${len_path_value_rels}))
      message(SEND_ERROR "PATHS lengths got confused")
    endif()

    # Transform variables mentioned in ARG_PATHS to relative paths for
    # the configure_file() call. Variables are copied to subscopeds by cmake,
    # so this only modifies the local copy of the variables.
    math(EXPR arg_path_limit "${len_paths} - 1")
    foreach(i RANGE ${arg_path_limit})
      list(GET ARG_PATHS ${i} val1)
      list(GET ARG_PATH_VALUES_RELATIVE ${i} val2)
      set(${val1} ${val2})
    endforeach()
  endif()

  configure_file(${site_in} ${site_out} @ONLY)

  if (EXISTS "${ARG_MAIN_CONFIG}")
    # Remember main config / generated site config for llvm-lit.in.
    get_property(LLVM_LIT_CONFIG_FILES GLOBAL PROPERTY LLVM_LIT_CONFIG_FILES)
    list(APPEND LLVM_LIT_CONFIG_FILES "${ARG_MAIN_CONFIG}" "${site_out}")
    set_property(GLOBAL PROPERTY LLVM_LIT_CONFIG_FILES ${LLVM_LIT_CONFIG_FILES})
  endif()
endfunction()

function(dump_all_cmake_variables)
  get_cmake_property(_variableNames VARIABLES)
  foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

function(get_llvm_lit_path base_dir file_name)
  cmake_parse_arguments(ARG "ALLOW_EXTERNAL" "" "" ${ARGN})

  if (ARG_ALLOW_EXTERNAL)
    set (LLVM_EXTERNAL_LIT "" CACHE STRING "Command used to spawn lit")
    if ("${LLVM_EXTERNAL_LIT}" STREQUAL "")
      set(LLVM_EXTERNAL_LIT "${LLVM_DEFAULT_EXTERNAL_LIT}")
    endif()

    if (NOT "${LLVM_EXTERNAL_LIT}" STREQUAL "")
      if (EXISTS ${LLVM_EXTERNAL_LIT})
        get_filename_component(LIT_FILE_NAME ${LLVM_EXTERNAL_LIT} NAME)
        get_filename_component(LIT_BASE_DIR ${LLVM_EXTERNAL_LIT} DIRECTORY)
        set(${file_name} ${LIT_FILE_NAME} PARENT_SCOPE)
        set(${base_dir} ${LIT_BASE_DIR} PARENT_SCOPE)
        return()
      elseif (NOT DEFINED CACHE{LLVM_EXTERNAL_LIT_MISSING_WARNED_ONCE})
        message(WARNING "LLVM_EXTERNAL_LIT set to ${LLVM_EXTERNAL_LIT}, but the path does not exist.")
        set(LLVM_EXTERNAL_LIT_MISSING_WARNED_ONCE YES CACHE INTERNAL "")
      endif()
    endif()
  endif()

  set(lit_file_name "llvm-lit")
  if (CMAKE_HOST_WIN32 AND NOT CYGWIN)
    # llvm-lit needs suffix.py for multiprocess to find a main module.
    set(lit_file_name "${lit_file_name}.py")
  endif ()
  set(${file_name} ${lit_file_name} PARENT_SCOPE)

  get_property(LLVM_LIT_BASE_DIR GLOBAL PROPERTY LLVM_LIT_BASE_DIR)
  if (NOT "${LLVM_LIT_BASE_DIR}" STREQUAL "")
    set(${base_dir} ${LLVM_LIT_BASE_DIR} PARENT_SCOPE)
  endif()

  # Allow individual projects to provide an override
  if (NOT "${LLVM_LIT_OUTPUT_DIR}" STREQUAL "")
    set(LLVM_LIT_BASE_DIR ${LLVM_LIT_OUTPUT_DIR})
  elseif(NOT "${LLVM_RUNTIME_OUTPUT_INTDIR}" STREQUAL "")
    set(LLVM_LIT_BASE_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  else()
    set(LLVM_LIT_BASE_DIR "")
  endif()

  # Cache this so we don't have to do it again and have subsequent calls
  # potentially disagree on the value.
  set_property(GLOBAL PROPERTY LLVM_LIT_BASE_DIR ${LLVM_LIT_BASE_DIR})
  set(${base_dir} ${LLVM_LIT_BASE_DIR} PARENT_SCOPE)
endfunction()

# A raw function to create a lit target. This is used to implement the testuite
# management functions.
function(add_lit_target target comment)
  cmake_parse_arguments(ARG "" "" "PARAMS;DEPENDS;ARGS" ${ARGN})
  set(LIT_ARGS "${ARG_ARGS} ${LLVM_LIT_ARGS}")
  separate_arguments(LIT_ARGS)
  if (NOT CMAKE_CFG_INTDIR STREQUAL ".")
    list(APPEND LIT_ARGS --param build_mode=${CMAKE_CFG_INTDIR})
  endif ()

  # Get the path to the lit to *run* tests with.  This can be overriden by
  # the user by specifying -DLLVM_EXTERNAL_LIT=<path-to-lit.py>
  get_llvm_lit_path(
    lit_base_dir
    lit_file_name
    ALLOW_EXTERNAL
    )

  set(LIT_COMMAND "${Python3_EXECUTABLE};${lit_base_dir}/${lit_file_name}")
  list(APPEND LIT_COMMAND ${LIT_ARGS})
  foreach(param ${ARG_PARAMS})
    list(APPEND LIT_COMMAND --param ${param})
  endforeach()
  if (ARG_UNPARSED_ARGUMENTS)
    add_custom_target(${target}
      COMMAND ${LIT_COMMAND} ${ARG_UNPARSED_ARGUMENTS}
      COMMENT "${comment}"
      USES_TERMINAL
      )
  else()
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, no tools built.")
    message(STATUS "${target} does nothing.")
  endif()
  get_subproject_title(subproject_title)
  set_target_properties(${target} PROPERTIES FOLDER "${subproject_title}/Tests")

  if (ARG_DEPENDS)
    add_dependencies(${target} ${ARG_DEPENDS})
  endif()

  # Tests should be excluded from "Build Solution".
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
  set_target_properties(${target} PROPERTIES XCODE_GENERATE_SCHEME ON)
endfunction()

# Convert a target name like check-clang to a variable name like CLANG.
function(umbrella_lit_testsuite_var target outvar)
  if (NOT target MATCHES "^check-")
    message(FATAL_ERROR "umbrella lit suites must be check-*, not '${target}'")
  endif()
  string(SUBSTRING "${target}" 6 -1 var)
  string(REPLACE "-" "_" var ${var})
  string(TOUPPER "${var}" var)
  set(${outvar} "${var}" PARENT_SCOPE)
endfunction()

# Start recording all lit test suites for a combined 'check-foo' target.
# The recording continues until umbrella_lit_testsuite_end() creates the target.
function(umbrella_lit_testsuite_begin target)
  umbrella_lit_testsuite_var(${target} name)
  set_property(GLOBAL APPEND PROPERTY LLVM_LIT_UMBRELLAS ${name})
endfunction()

# Create a combined 'check-foo' target for a set of related test suites.
# It runs all suites added since the matching umbrella_lit_testsuite_end() call.
# Tests marked EXCLUDE_FROM_CHECK_ALL are not gathered.
function(umbrella_lit_testsuite_end target)
  umbrella_lit_testsuite_var(${target} name)

  get_property(testsuites GLOBAL PROPERTY LLVM_${name}_LIT_TESTSUITES)
  get_property(params GLOBAL PROPERTY LLVM_${name}_LIT_PARAMS)
  get_property(depends GLOBAL PROPERTY LLVM_${name}_LIT_DEPENDS)
  get_property(extra_args GLOBAL PROPERTY LLVM_${name}_LIT_EXTRA_ARGS)
  # Additional test targets are not gathered, but may be set externally.
  get_property(additional_test_targets
               GLOBAL PROPERTY LLVM_${name}_ADDITIONAL_TEST_TARGETS)

  string(TOLOWER ${name} name)
  add_lit_target(${target}
    "Running ${name} regression tests"
    ${testsuites}
    PARAMS ${params}
    DEPENDS ${depends} ${additional_test_targets}
    ARGS ${extra_args}
    )
endfunction()

# A function to add a set of lit test suites to be driven through 'check-*' targets.
function(add_lit_testsuite target comment)
  cmake_parse_arguments(ARG "EXCLUDE_FROM_CHECK_ALL" "" "PARAMS;DEPENDS;ARGS" ${ARGN})

  # EXCLUDE_FROM_ALL excludes the test ${target} out of check-all.
  if(NOT ARG_EXCLUDE_FROM_CHECK_ALL)
    get_property(gather_names GLOBAL PROPERTY LLVM_LIT_UMBRELLAS)
    foreach(name ${gather_names})
    # Register the testsuites, params and depends for the umbrella check rule.
      set_property(GLOBAL APPEND PROPERTY LLVM_${name}_LIT_TESTSUITES ${ARG_UNPARSED_ARGUMENTS})
      set_property(GLOBAL APPEND PROPERTY LLVM_${name}_LIT_PARAMS ${ARG_PARAMS})
      set_property(GLOBAL APPEND PROPERTY LLVM_${name}_LIT_DEPENDS ${ARG_DEPENDS})
      set_property(GLOBAL APPEND PROPERTY LLVM_${name}_LIT_EXTRA_ARGS ${ARG_ARGS})
    endforeach()
  endif()

  # Produce a specific suffixed check rule.
  add_lit_target(${target} ${comment}
    ${ARG_UNPARSED_ARGUMENTS}
    PARAMS ${ARG_PARAMS}
    DEPENDS ${ARG_DEPENDS}
    ARGS ${ARG_ARGS}
    )
endfunction()

function(add_lit_testsuites project directory)
  if (NOT LLVM_ENABLE_IDE)
    cmake_parse_arguments(ARG
      "EXCLUDE_FROM_CHECK_ALL"
      "FOLDER;BINARY_DIR"
      "PARAMS;DEPENDS;ARGS;SKIP"
      ${ARGN}
      )

    if (NOT ARG_FOLDER)
      get_subproject_title(subproject_title)
      set(ARG_FOLDER "${subproject_title}/Tests/LIT Testsuites")
    endif()

    # Search recursively for test directories by assuming anything not
    # in a directory called Inputs contains tests.
    file(GLOB_RECURSE to_process LIST_DIRECTORIES true ${directory}/*)
    foreach(lit_suite ${to_process})
      if(NOT IS_DIRECTORY ${lit_suite})
        continue()
      endif()
      string(FIND ${lit_suite} Inputs is_inputs)
      string(FIND ${lit_suite} Output is_output)
      if (NOT (is_inputs EQUAL -1 AND is_output EQUAL -1))
        continue()
      endif()

      # Create a check- target for the directory.
      string(REPLACE "${directory}/" "" name_slash ${lit_suite})
      set(_should_skip FALSE)
      foreach(skip IN LISTS ARG_SKIP)
        if(name_slash MATCHES "${skip}")
          set(_should_skip TRUE)
          break()
        endif()
      endforeach()
      if (_should_skip)
        continue()
      endif()
      if (name_slash)
        set(filter ${name_slash})
        string(REPLACE "/" "-" name_slash ${name_slash})
        string(REPLACE "\\" "-" name_dashes ${name_slash})
        string(TOLOWER "${project}-${name_dashes}" name_var)
        set(lit_args ${lit_suite})
        if (ARG_BINARY_DIR)
          set(lit_args ${ARG_BINARY_DIR} --filter=${filter})
        endif()
        add_lit_target("check-${name_var}" "Running lit suite ${lit_suite}"
          ${lit_args}
          ${EXCLUDE_FROM_CHECK_ALL}
          PARAMS ${ARG_PARAMS}
          DEPENDS ${ARG_DEPENDS}
          ARGS ${ARG_ARGS}
        )
        set_target_properties(check-${name_var} PROPERTIES FOLDER ${ARG_FOLDER})
      endif()
    endforeach()
  endif()
endfunction()
