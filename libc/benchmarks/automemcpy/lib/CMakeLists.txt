add_library(automemcpy_codegen CodeGen.cpp)
target_link_libraries(automemcpy_codegen PUBLIC LLVMSupport)
target_include_directories(automemcpy_codegen PUBLIC ${LIBC_AUTOMEMCPY_INCLUDE_DIR})
llvm_update_compile_flags(automemcpy_codegen)

add_executable(automemcpy_codegen_main CodeGenMain.cpp RandomFunctionGenerator.cpp)
target_link_libraries(automemcpy_codegen_main PUBLIC automemcpy_codegen ${Z3_LIBRARIES})
llvm_update_compile_flags(automemcpy_codegen_main)

set(Implementations "${CMAKE_CURRENT_BINARY_DIR}/Implementations.cpp")
add_custom_command(
    OUTPUT ${Implementations}
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/automemcpy_codegen_main" > "${Implementations}"
    COMMAND echo "automemcpy implementations generated in ${Implementations}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS automemcpy_codegen_main
)

add_library(automemcpy_implementations "${Implementations}")
target_link_libraries(automemcpy_implementations PUBLIC LLVMSupport libc-memory-benchmark)
target_include_directories(automemcpy_implementations PRIVATE ${LIBC_SOURCE_DIR} ${LIBC_AUTOMEMCPY_INCLUDE_DIR})
target_compile_options(automemcpy_implementations PRIVATE ${LIBC_COMPILE_OPTIONS_NATIVE} "SHELL:-mllvm -combiner-global-alias-analysis" -fno-builtin)
llvm_update_compile_flags(automemcpy_implementations)

add_executable(automemcpy EXCLUDE_FROM_ALL ${LIBC_SOURCE_DIR}/benchmarks/LibcMemoryGoogleBenchmarkMain.cpp)
target_link_libraries(automemcpy PRIVATE libc-memory-benchmark benchmark_main automemcpy_implementations)
llvm_update_compile_flags(automemcpy)

add_library(automemcpy_result_analyzer_lib EXCLUDE_FROM_ALL ResultAnalyzer.cpp)
target_link_libraries(automemcpy_result_analyzer_lib PUBLIC LLVMSupport)
target_include_directories(automemcpy_result_analyzer_lib PUBLIC ${LIBC_AUTOMEMCPY_INCLUDE_DIR})
llvm_update_compile_flags(automemcpy_result_analyzer_lib)

add_executable(automemcpy_result_analyzer EXCLUDE_FROM_ALL ResultAnalyzerMain.cpp)
target_link_libraries(automemcpy_result_analyzer PRIVATE automemcpy_result_analyzer_lib automemcpy_implementations)
llvm_update_compile_flags(automemcpy_result_analyzer)
