add_custom_target(FlangUnitTests)
set_target_properties(FlangUnitTests PROPERTIES FOLDER "Flang Unit Tests")

function(add_flang_unittest test_dirname)
  add_unittest(FlangUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Optimizer)
add_subdirectory(Decimal)
add_subdirectory(Evaluate)
add_subdirectory(Runtime)
add_subdirectory(Lower)
