add_custom_target(ExtraToolsUnitTests)
set_target_properties(ExtraToolsUnitTests PROPERTIES FOLDER "Extra Tools Unit Tests")

function(add_extra_unittest test_dirname)
  add_unittest(ExtraToolsUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(clang-apply-replacements)
add_subdirectory(clang-rename)
add_subdirectory(clang-query)
add_subdirectory(clang-tidy)
