# Run the Completion Model Codegenerator on the model present in the 
# ${model} directory.
# Produces a pair of files called ${filename}.h and  ${filename}.cpp in the 
# ${CMAKE_CURRENT_BINARY_DIR}. The generated header
# will define a C++ class called ${cpp_class} - which may be a
# namespace-qualified class name.
function(gen_decision_forest model filename cpp_class)
  set(model_compiler ${CMAKE_SOURCE_DIR}/../clang-tools-extra/clangd/quality/CompletionModelCodegen.py)
  
  set(output_dir ${CMAKE_CURRENT_BINARY_DIR})
  set(header_file ${output_dir}/${filename}.h)
  set(cpp_file ${output_dir}/${filename}.cpp)

  add_custom_command(OUTPUT ${header_file} ${cpp_file}
    COMMAND "${Python3_EXECUTABLE}" ${model_compiler}
      --model ${model}
      --output_dir ${output_dir}
      --filename ${filename}
      --cpp_class ${cpp_class}
    COMMENT "Generating code completion model runtime..."
    DEPENDS ${model_compiler} ${model}/forest.json ${model}/features.json
    VERBATIM )

  set_source_files_properties(${header_file} PROPERTIES
    GENERATED 1)
  set_source_files_properties(${cpp_file} PROPERTIES
    GENERATED 1)

  # Disable unused label warning for generated files.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_source_files_properties(${cpp_file} PROPERTIES
      COMPILE_FLAGS /wd4102)
  else()
    set_source_files_properties(${cpp_file} PROPERTIES
      COMPILE_FLAGS -Wno-unused)
  endif()
endfunction()
