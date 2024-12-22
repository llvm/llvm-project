add_libc_benchmark_unittest(libc-automemcpy-codegen-test
    SRCS CodeGenTest.cpp
    DEPENDS automemcpy_codegen
)

add_libc_benchmark_unittest(libc-automemcpy-result-analyzer-test
    SRCS ResultAnalyzerTest.cpp
    DEPENDS automemcpy_result_analyzer_lib
)
