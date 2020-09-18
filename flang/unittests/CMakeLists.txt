add_custom_target(FlangUnitTests)
set_target_properties(FlangUnitTests PROPERTIES FOLDER "Flang Unit Tests")

function(add_flang_unittest test_dirname)
  add_unittest(FlangUnitTests ${test_dirname} ${ARGN})
endfunction()

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()

function(add_flang_nongtest_unittest test_name)
  add_executable(${test_name}.test ${test_name}.cpp)

  target_link_libraries(${test_name}.test ${ARGN})

  add_dependencies(FlangUnitTests ${test_name}.test)
endfunction()

add_subdirectory(Optimizer)
add_subdirectory(Decimal)
add_subdirectory(Evaluate)
add_subdirectory(Runtime)
add_subdirectory(Lower)

if (FLANG_BUILD_NEW_DRIVER)
  add_subdirectory(Frontend)
endif()
