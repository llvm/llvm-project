# Sort a subset of a list according to the ordering in the full list.
#
# Given a list and a subset of that list, this function sorts the subset
# according to the order in the full list, and returns that in the given
# output variable.
#
# full_list:
#   The list containing the desired order of elements in the sub-list.
#
# sub_list:
#   A subset of the elements in `full_list`. Those elements will be sorted
#   according to the order in `full_list`.
#
# out_var:
#   A variable to store the resulting sorted sub-list in.
function(sort_subset full_list sub_list out_var)
  set(result "${full_list}")
  foreach(project IN LISTS full_list)
    if (NOT project IN_LIST sub_list)
      list(REMOVE_ITEM result ${project})
    endif()
  endforeach()

  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()
