set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_library(FortranEvaluateTesting
  testing.cpp
  fp-testing.cpp
)

add_executable(leading-zero-bit-count-test
  leading-zero-bit-count.cpp
)

target_link_libraries(leading-zero-bit-count-test
  FortranEvaluateTesting
  LLVMSupport
)

add_executable(bit-population-count-test
  bit-population-count.cpp
)

target_link_libraries(bit-population-count-test
  FortranEvaluateTesting
  LLVMSupport
)

add_executable(uint128-test
  uint128.cpp
)

target_link_libraries(uint128-test
  FortranEvaluateTesting
  LLVMSupport
)

# These routines live in lib/Common but we test them here.
add_test(UINT128 uint128-test)
add_test(Leadz leading-zero-bit-count-test)
add_test(PopPar bit-population-count-test)

add_executable(expression-test
  expression.cpp
)

target_link_libraries(expression-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  FortranParser
  LLVMSupport
)

add_executable(integer-test
  integer.cpp
)

target_link_libraries(integer-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  LLVMSupport
)

add_executable(intrinsics-test
  intrinsics.cpp
)

target_link_libraries(intrinsics-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranDecimal
  FortranSemantics
  FortranParser
  FortranRuntime
  LLVMSupport
)

add_executable(logical-test
  logical.cpp
)

target_link_libraries(logical-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  LLVMSupport
)

# GCC -fno-exceptions breaks the fenv.h interfaces needed to capture
# IEEE exception flags (different use of the word "exception")
# in the actual hardware floating-point status register, so ensure that
# C++ exceptions are enabled for this test.
set_source_files_properties(real.cpp PROPERTIES COMPILE_FLAGS -fexceptions)
add_executable(real-test
  real.cpp
)

target_link_libraries(real-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranDecimal
  FortranSemantics
  LLVMSupport
)

add_executable(reshape-test
  reshape.cpp
)

target_link_libraries(reshape-test
  FortranEvaluateTesting
  FortranSemantics
  FortranEvaluate
  FortranRuntime
  LLVMSupport
)

add_executable(ISO-Fortran-binding-test
  ISO-Fortran-binding.cpp
)

target_link_libraries(ISO-Fortran-binding-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  FortranRuntime
  LLVMSupport
)

add_executable(folding-test
  folding.cpp
)

target_link_libraries(folding-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  LLVMSupport
)

add_test(Expression expression-test)
add_test(Integer integer-test)
add_test(Intrinsics intrinsics-test)
add_test(Logical logical-test)
add_test(Real real-test)
add_test(RESHAPE reshape-test)
add_test(ISO-binding ISO-Fortran-binding-test)
add_test(folding folding-test)
