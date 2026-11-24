// This file intentionally contains no test cases. CMake's add_executable
// requires a source file, but successfully linking the executable is the test.
// When LLVM_LIBC_HERMETIC_TEST_USE_INTERNAL_STARTUP is disabled, a hermetic
// test must link the minimal external startup object supplied by this test and
// must not link LLVM libc's internal crt1. Both startup objects define _start,
// so accidentally linking the internal crt1 as well causes a duplicate-symbol
// error. A successful link therefore verifies that disabling internal startup
// selects only the provided minimal_external_crt1_for_test.cpp object.
