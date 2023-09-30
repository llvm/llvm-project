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

add_flang_nongtest_unittest(leading-zero-bit-count
  FortranEvaluateTesting
)

add_flang_nongtest_unittest(bit-population-count
  FortranEvaluateTesting
)

add_flang_nongtest_unittest(uint128
  FortranEvaluateTesting
)

add_flang_nongtest_unittest(expression
  FortranCommon
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  FortranParser
)

add_flang_nongtest_unittest(integer
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
)

add_flang_nongtest_unittest(intrinsics
  FortranCommon
  FortranEvaluateTesting
  FortranEvaluate
  FortranDecimal
  FortranSemantics
  FortranParser
)

add_flang_nongtest_unittest(logical
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
add_flang_nongtest_unittest(real
  FortranEvaluateTesting
  FortranEvaluate
  FortranDecimal
  FortranSemantics
)
llvm_update_compile_flags(real.test)

add_flang_nongtest_unittest(folding
  FortranCommon
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
)
