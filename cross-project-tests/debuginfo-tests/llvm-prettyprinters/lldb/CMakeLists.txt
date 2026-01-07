# Unset _FORTIFY_SOURCE since it's incompatible with -O0.
set(LLDB_TEST_DEBUG_COMPILE_OPTIONS -g -O0 -U_FORTIFY_SOURCE)

add_executable(check-lldb-llvm-support-arrayref arrayref.cpp)
target_link_libraries(check-lldb-llvm-support-arrayref PRIVATE LLVMSupport)
target_compile_options(check-lldb-llvm-support-arrayref PRIVATE ${LLDB_TEST_DEBUG_COMPILE_OPTIONS})

set(LLDB_FORMATTER_TESTS
    check-lldb-llvm-support-arrayref
    PARENT_SCOPE)
