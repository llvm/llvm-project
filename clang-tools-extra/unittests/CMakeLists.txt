add_custom_target(ExtraToolsUnitTests)
set_target_properties(ExtraToolsUnitTests PROPERTIES FOLDER "Clang Tools Extra/Tests")

function(add_extra_unittest test_dirname)
  add_unittest(ExtraToolsUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(clang-apply-replacements)
add_subdirectory(clang-change-namespace)
add_subdirectory(clang-doc)
add_subdirectory(clang-include-fixer)
add_subdirectory(clang-move)
add_subdirectory(clang-query)
add_subdirectory(clang-tidy)
