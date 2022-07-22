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

# Special flags
set(FMA_OPT_FLAG "FMA_OPT")
set(ROUND_OPT_FLAG "ROUND_OPT")

# Skip FMA_OPT flag for targets that don't support fma.
if(NOT(LIBC_TARGET_ARCHITECTURE_IS_X86 AND (LIBC_CPU_FEATURES MATCHES "FMA")))
  set(SKIP_FLAG_EXPANSION_FMA_OPT TRUE)
endif()

# Skip ROUND_OPT flag for targets that don't support SSE 4.2.
if(NOT(LIBC_TARGET_ARCHITECTURE_IS_X86 AND (LIBC_CPU_FEATURES MATCHES "SSE4_2")))
  set(SKIP_FLAG_EXPANSION_ROUND_OPT TRUE)
endif()
