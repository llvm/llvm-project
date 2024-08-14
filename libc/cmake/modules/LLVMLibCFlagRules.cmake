# In general, a flag is a string provided for supported functions under the
# multi-valued option `FLAGS`.  It should be one of the following forms:
#   FLAG_NAME
#   FLAG_NAME__NO
#   FLAG_NAME__ONLY
# A target will inherit all the flags of its upstream dependency.
#
# When we create a target `TARGET_NAME` with a flag using (add_header_library,
# add_object_library, ...), its behavior will depend on the flag form as follow:
# - FLAG_NAME: The following 2 targets will be generated:
#     `TARGET_NAME` that has `FLAG_NAME` in its `FLAGS` property.
#     `TARGET_NAME.__NO_FLAG_NAME` that depends on `DEP.__NO_FLAG_NAME` if
#        `TARGET_NAME` depends on `DEP` and `DEP` has `FLAG_NAME` in its `FLAGS`
#        property.
# - FLAG_NAME__ONLY: Only generate 1 target `TARGET_NAME` that has `FLAG_NAME`
#     in its `FLAGS` property.
# - FLAG_NAME__NO: Only generate 1 target `TARGET_NAME` that depends on
# `DEP.__NO_FLAG_NAME` if `DEP` is in its DEPENDS list and `DEP` has `FLAG_NAME`
# in its `FLAGS` property.
#
# To show all the targets generated, pass SHOW_INTERMEDIATE_OBJECTS=ON to cmake.
# To show all the targets' dependency and flags, pass
#   SHOW_INTERMEDIATE_OBJECTS=DEPS to cmake.
#
# To completely disable a flag FLAG_NAME expansion, set the variable
#   SKIP_FLAG_EXPANSION_FLAG_NAME=TRUE in this file.


function(extract_flag_modifier input_flag output_flag modifier)
  if(${input_flag} MATCHES "__NO$")
    string(REGEX REPLACE "__NO$" "" flag "${input_flag}")
    set(${output_flag} ${flag} PARENT_SCOPE)
    set(${modifier} "NO" PARENT_SCOPE)
  elseif(${input_flag} MATCHES "__ONLY$")
    string(REGEX REPLACE "__ONLY$" "" flag "${input_flag}")
    set(${output_flag} ${flag} PARENT_SCOPE)
    set(${modifier} "ONLY" PARENT_SCOPE)
  else()
    set(${output_flag} ${input_flag} PARENT_SCOPE)
    set(${modifier} "" PARENT_SCOPE)
  endif()
endfunction(extract_flag_modifier)

function(remove_duplicated_flags input_flags output_flags)
  set(out_flags "")
  foreach(input_flag IN LISTS input_flags)
    if(NOT input_flag)
      continue()
    endif()

    extract_flag_modifier(${input_flag} flag modifier)

    # Check if the flag is skipped.
    if(${SKIP_FLAG_EXPANSION_${flag}})
      if("${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS")
        message(STATUS "  Flag ${flag} is ignored.")
      endif()
      continue()
    endif()

    set(found FALSE)
    foreach(out_flag IN LISTS out_flags)
      extract_flag_modifier(${out_flag} o_flag o_modifier)
      if("${flag}" STREQUAL "${o_flag}")
        set(found TRUE)
        break()
      endif()
    endforeach()
    if(NOT found)
      list(APPEND out_flags ${input_flag})
    endif()
  endforeach()

  set(${output_flags} "${out_flags}" PARENT_SCOPE)
endfunction(remove_duplicated_flags)

# Collect flags from dependency list.  To see which flags come with each
# dependence, pass `SHOW_INTERMEDIATE_OBJECTS=DEPS` to cmake.
function(get_flags_from_dep_list output_list)
  set(flag_list "")
  foreach(dep IN LISTS ARGN)
    if(NOT dep)
      continue()
    endif()

    get_fq_dep_name(fq_dep_name ${dep})

    if(NOT TARGET ${fq_dep_name})
      continue()
    endif()

    get_target_property(flags ${fq_dep_name} "FLAGS")

    if(flags AND "${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS")
      message(STATUS "  FLAGS from dependency ${fq_dep_name} are ${flags}")
    endif()

    foreach(flag IN LISTS flags)
      if(flag)
        list(APPEND flag_list ${flag})
      endif()
    endforeach()
  endforeach(dep)

  list(REMOVE_DUPLICATES flag_list)

  set(${output_list} ${flag_list} PARENT_SCOPE)
endfunction(get_flags_from_dep_list)

# Given a `flag` without modifier, scan through the list of dependency, append
# `.__NO_flag` to any target that has `flag` in its FLAGS property.
function(get_fq_dep_list_without_flag output_list flag)
  set(fq_dep_no_flag_list "")
  foreach(dep IN LISTS ARGN)
    get_fq_dep_name(fq_dep_name ${dep})
    if(TARGET ${fq_dep_name})
      get_target_property(dep_flags ${fq_dep_name} "FLAGS")
      # Only target with `flag` has `.__NO_flag` target, `flag__NO` and
      # `flag__ONLY` do not.
      if(${flag} IN_LIST dep_flags)
        list(APPEND fq_dep_no_flag_list "${fq_dep_name}.__NO_${flag}")
      else()
        list(APPEND fq_dep_no_flag_list ${fq_dep_name})
      endif()
    else()
      list(APPEND fq_dep_no_flag_list ${fq_dep_name})
    endif()
  endforeach(dep)
  set(${output_list} ${fq_dep_no_flag_list} PARENT_SCOPE)
endfunction(get_fq_dep_list_without_flag)

# Check if a `flag` is set
function(check_flag result flag_name)
  list(FIND ARGN ${flag_name} has_flag)
  if(${has_flag} LESS 0)
    list(FIND ARGN "${flag_name}__ONLY" has_flag)
  endif()
  if(${has_flag} GREATER -1)
    set(${result} TRUE PARENT_SCOPE)
  else()
    set(${result} FALSE PARENT_SCOPE)
  endif()
endfunction(check_flag)

# Generate all flags' combinations and call the corresponding function provided
# by `CREATE_TARGET` to create a target for each combination.
function(expand_flags_for_target target_name flags)
  cmake_parse_arguments(
    "EXPAND_FLAGS"
    "" # Optional arguments
    "CREATE_TARGET" # Single-value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  list(LENGTH flags nflags)
  if(NOT ${nflags})
    cmake_language(CALL ${EXPAND_FLAGS_CREATE_TARGET}
      ${target_name}
      ${EXPAND_FLAGS_UNPARSED_ARGUMENTS}
      DEPENDS ${EXPAND_FLAGS_DEPENDS}
      FLAGS ${EXPAND_FLAGS_FLAGS}
    )
    return()
  endif()

  list(GET flags 0 flag)
  list(REMOVE_AT flags 0)
  extract_flag_modifier(${flag} real_flag modifier)

  if(NOT "${modifier}" STREQUAL "NO")
    expand_flags_for_target(
      ${target_name}
      "${flags}"
      DEPENDS ${EXPAND_FLAGS_DEPENDS}
      FLAGS ${EXPAND_FLAGS_FLAGS}
      CREATE_TARGET ${EXPAND_FLAGS_CREATE_TARGET}
      ${EXPAND_FLAGS_UNPARSED_ARGUMENTS}
    )
  endif()

  if("${real_flag}" STREQUAL "" OR "${modifier}" STREQUAL "ONLY")
    return()
  endif()

  set(NEW_FLAGS ${EXPAND_FLAGS_FLAGS})
  list(REMOVE_ITEM NEW_FLAGS ${flag})
  get_fq_dep_list_without_flag(NEW_DEPS ${real_flag} ${EXPAND_FLAGS_DEPENDS})

  # Only target with `flag` has `.__NO_flag` target, `flag__NO` and
  # `flag__ONLY` do not.
  if("${modifier}" STREQUAL "")
    set(TARGET_NAME "${target_name}.__NO_${flag}")
  else()
    set(TARGET_NAME "${target_name}")
  endif()

  expand_flags_for_target(
    ${TARGET_NAME}
    "${flags}"
    DEPENDS ${NEW_DEPS}
    FLAGS ${NEW_FLAGS}
    CREATE_TARGET ${EXPAND_FLAGS_CREATE_TARGET}
    ${EXPAND_FLAGS_UNPARSED_ARGUMENTS}
  )
endfunction(expand_flags_for_target)

# Collect all flags from a target's dependency, and then forward to
# `expand_flags_for_target to generate all flags' combinations and call
# the corresponding function provided by `CREATE_TARGET` to create a target for
# each combination.
function(add_target_with_flags target_name)
  cmake_parse_arguments(
    "ADD_TO_EXPAND"
    "" # Optional arguments
    "CREATE_TARGET;" # Single value arguments
    "DEPENDS;FLAGS;ADD_FLAGS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT target_name)
    message(FATAL_ERROR "Bad target name")
  endif()

  if(NOT ADD_TO_EXPAND_CREATE_TARGET)
    message(FATAL_ERROR "Missing function to create targets.  Please specify "
                        "`CREATE_TARGET <function>`")
  endif()

  get_fq_target_name(${target_name} fq_target_name)

  if(ADD_TO_EXPAND_DEPENDS AND ("${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS"))
    message(STATUS "Gathering FLAGS from dependencies for ${fq_target_name}")
  endif()

  get_fq_deps_list(fq_deps_list ${ADD_TO_EXPAND_DEPENDS})
  get_flags_from_dep_list(deps_flag_list ${fq_deps_list})

  # Appending ADD_FLAGS before flags from dependency.
  if(ADD_TO_EXPAND_ADD_FLAGS)
    list(APPEND ADD_TO_EXPAND_FLAGS ${ADD_TO_EXPAND_ADD_FLAGS})
  endif()
  list(APPEND ADD_TO_EXPAND_FLAGS ${deps_flag_list})
  remove_duplicated_flags("${ADD_TO_EXPAND_FLAGS}" flags)
  list(SORT flags)

  if(SHOW_INTERMEDIATE_OBJECTS AND flags)
    message(STATUS "Target ${fq_target_name} has FLAGS: ${flags}")
  endif()

  expand_flags_for_target(
    ${fq_target_name}
    "${flags}"
    DEPENDS "${fq_deps_list}"
    FLAGS "${flags}"
    CREATE_TARGET ${ADD_TO_EXPAND_CREATE_TARGET}
    ${ADD_TO_EXPAND_UNPARSED_ARGUMENTS}
  )
endfunction(add_target_with_flags)

# Special flags
set(FMA_OPT_FLAG "FMA_OPT")
set(ROUND_OPT_FLAG "ROUND_OPT")
# This flag controls whether we use explicit SIMD instructions or not.
set(EXPLICIT_SIMD_OPT_FLAG "EXPLICIT_SIMD_OPT")
# This flag controls whether we use compiler builtin functions to implement
# various basic math operations or not.
set(MISC_MATH_BASIC_OPS_OPT_FLAG "MISC_MATH_BASIC_OPS_OPT")

# Skip FMA_OPT flag for targets that don't support fma.
if(NOT((LIBC_TARGET_ARCHITECTURE_IS_X86 AND (LIBC_CPU_FEATURES MATCHES "FMA")) OR
       LIBC_TARGET_ARCHITECTURE_IS_RISCV64))
  set(SKIP_FLAG_EXPANSION_FMA_OPT TRUE)
endif()

# Skip EXPLICIT_SIMD_OPT flag for targets that don't support SSE2.
# Note: one may want to revisit it if they want to control other explicit SIMD
if(NOT(LIBC_TARGET_ARCHITECTURE_IS_X86 AND (LIBC_CPU_FEATURES MATCHES "SSE2")))
  set(SKIP_FLAG_EXPANSION_EXPLICIT_SIMD_OPT TRUE)
endif()

# Skip ROUND_OPT flag for targets that don't support rounding instructions. On
# x86, these are SSE4.1 instructions, but we already had code to check for
# SSE4.2 support.
if(NOT((LIBC_TARGET_ARCHITECTURE_IS_X86 AND (LIBC_CPU_FEATURES MATCHES "SSE4_2")) OR
       LIBC_TARGET_ARCHITECTURE_IS_AARCH64 OR LIBC_TARGET_OS_IS_GPU))
  set(SKIP_FLAG_EXPANSION_ROUND_OPT TRUE)
endif()

# Choose whether time_t is 32- or 64-bit, based on target architecture
# and config options. This will be used to set a #define during the
# library build, and also to select the right version of time_t.h for
# the output headers.
if(LIBC_TARGET_ARCHITECTURE_IS_ARM AND NOT (LIBC_CONF_TIME_64BIT))
  # Set time_t to 32 bit for compatibility with glibc, unless
  # configuration says otherwise
  set(LIBC_TYPES_TIME_T_IS_32_BIT TRUE)
else()
  # Other platforms default to 64-bit time_t
  set(LIBC_TYPES_TIME_T_IS_32_BIT FALSE)
endif()
