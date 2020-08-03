set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_library(FortranEvaluateTesting
  testing.cpp
  fp-testing.cpp
)
if (LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs Support)
endif()
target_link_libraries(FortranEvaluateTesting
    ${llvm_libs})

add_executable(leading-zero-bit-count-test
  leading-zero-bit-count.cpp
)

target_link_libraries(leading-zero-bit-count-test
  FortranEvaluateTesting
)

add_executable(bit-population-count-test
  bit-population-count.cpp
)

target_link_libraries(bit-population-count-test
  FortranEvaluateTesting
)

add_executable(uint128-test
  uint128.cpp
)

target_link_libraries(uint128-test
  FortranEvaluateTesting
)

# These routines live in lib/Common but we test them here.
add_test(UINT128 uint128-test)
add_test(Leadz leading-zero-bit-count-test)
add_test(PopPar bit-population-count-test)

add_executable(expression-test
  expression.cpp
)

target_link_libraries(expression-test
  FortranCommon
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  FortranParser
)

add_executable(integer-test
  integer.cpp
)

target_link_libraries(integer-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
)

add_executable(intrinsics-test
  intrinsics.cpp
)

target_link_libraries(intrinsics-test
  FortranCommon
  FortranEvaluateTesting
  FortranEvaluate
  FortranDecimal
  FortranSemantics
  FortranParser
  FortranRuntime
)

add_executable(logical-test
  logical.cpp
)

target_link_libraries(logical-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
)

# GCC -fno-exceptions breaks the fenv.h interfaces needed to capture
# IEEE exception flags (different use of the word "exception")
# in the actual hardware floating-point status register, so ensure that
# C++ exceptions are enabled for this test.
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
add_executable(real-test
  real.cpp
)
llvm_update_compile_flags(real-test)

target_link_libraries(real-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranDecimal
  FortranSemantics
)

add_executable(reshape-test
  reshape.cpp
)

target_link_libraries(reshape-test
  FortranEvaluateTesting
  FortranSemantics
  FortranEvaluate
  FortranRuntime
)

add_executable(ISO-Fortran-binding-test
  ISO-Fortran-binding.cpp
)

target_link_libraries(ISO-Fortran-binding-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  FortranRuntime
)

add_executable(folding-test
  folding.cpp
)

target_link_libraries(folding-test
  FortranCommon
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
)

add_test(Expression expression-test)
add_test(Integer integer-test)
add_test(Intrinsics intrinsics-test)
add_test(Logical logical-test)
add_test(Real real-test)
add_test(RESHAPE reshape-test)
add_test(ISO-binding ISO-Fortran-binding-test)
add_test(folding folding-test)
