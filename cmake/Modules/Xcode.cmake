# When this is enabled, CMake will generate schemes for every target, but not
# all of them make sense to surface in the Xcode UI by default.
set(CMAKE_XCODE_GENERATE_SCHEME YES)

# Enumerate all the targets in a directory.
macro(get_all_targets targets dir)
  get_property(sub_dirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(subdir ${sub_dirs})
    get_all_targets(${targets} ${subdir})
  endforeach()
  get_property(local_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
  list(APPEND ${targets} ${local_targets})
endmacro()

get_all_targets(all_targets ${PROJECT_SOURCE_DIR})

# Turn off scheme generation by default for targets that do not have
# XCODE_GENERATE_SCHEME set.
foreach(target ${all_targets})
  get_target_property(value ${target} XCODE_GENERATE_SCHEME)
  if("${value}" STREQUAL "value-NOTFOUND")
    set_target_properties(${target} PROPERTIES XCODE_GENERATE_SCHEME NO)
  endif()
endforeach()
