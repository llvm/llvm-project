add_custom_target(FlangUnitTests)
set_target_properties(FlangUnitTests PROPERTIES FOLDER "Flang Unit Tests")

function(add_flang_unittest test_dirname)
  add_unittest(FlangUnitTests ${test_dirname} ${ARGN})
endfunction()

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()

function(add_flang_nongtest_unittest test_name)
  cmake_parse_arguments(ARG
    "SLOW_TEST"
    ""
    ""
    ${ARGN})

  if(ARG_SLOW_TEST)
      set(suffix .slow)
  else()
      set(suffix .test)
  endif()

  add_executable(${test_name}${suffix} ${test_name}.cpp)

  if (LLVM_LINK_LLVM_DYLIB AND NOT ARG_DISABLE_LLVM_LINK_LLVM_DYLIB)
    set(llvm_libs LLVM)
  else()
    llvm_map_components_to_libnames(llvm_libs Support)
  endif()
  target_link_libraries(${test_name}${suffix} ${llvm_libs} ${ARG_UNPARSED_ARGUMENTS})

  if(NOT ARG_SLOW_TEST)
    add_dependencies(FlangUnitTests ${test_name}${suffix})
  endif()
endfunction()

add_subdirectory(Optimizer)
add_subdirectory(Common)
add_subdirectory(Decimal)
add_subdirectory(Evaluate)
add_subdirectory(Runtime)
add_subdirectory(Frontend)
