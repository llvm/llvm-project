add_library(lldb-formatter-test-settings INTERFACE)

target_link_libraries(lldb-formatter-test-settings INTERFACE LLVMSupport)
# Unset _FORTIFY_SOURCE since it's incompatible with -O0.
target_compile_options(lldb-formatter-test-settings INTERFACE -g -O0 -U_FORTIFY_SOURCE)

macro(add_lldb_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE lldb-formatter-test-settings)
endmacro()

add_lldb_test(check-lldb-llvm-support-arrayref arrayref.cpp)
add_lldb_test(check-lldb-llvm-support-pointer-int-pair pointer-int-pair.cpp)

set(LLDB_FORMATTER_TESTS
    check-lldb-llvm-support-arrayref
    check-lldb-llvm-support-pointer-int-pair
    PARENT_SCOPE)
