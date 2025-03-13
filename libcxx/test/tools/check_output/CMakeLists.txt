
add_executable(check_output check_output.cpp)

# put the binary into <build>/bin
set_target_properties(check_output PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${LIBCXX_BINARY_DIR}/bin")

# set the language mode
set_target_properties(check_output PROPERTIES
                      CXX_STANDARD 23
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)

target_compile_options(check_output PRIVATE -Werror=missing-prototypes)

add_dependencies(cxx-test-depends check_output)
