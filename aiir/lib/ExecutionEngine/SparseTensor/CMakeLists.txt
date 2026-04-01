# Unlike aiir_float16_utils, aiir_c_runner_utils, etc, we do *not* make
# this a shared library: because doing so causes issues when building
# on Windows.  In particular, various functions take/return `std::vector`
# but that class is not designed with dllimport/dllexport pragma,
# therefore it cannot be safely/correctly used across DLL boundaries.
# Consequently, we avoid using the "aiir_xxx_utils" naming scheme, since
# that is reserved/intended for shared libraries only.
add_aiir_library(AIIRSparseTensorRuntime
  File.cpp
  MapRef.cpp
  Storage.cpp

  EXCLUDE_FROM_LIBAIIR

  STANDALONE
  )
aiir_target_link_libraries(AIIRSparseTensorRuntime PUBLIC
  AIIRSparseTensorEnums
  aiir_float16_utils
  )
set_property(TARGET AIIRSparseTensorRuntime PROPERTY CXX_STANDARD 17)

check_cxx_compiler_flag(-Wweak-vtables
  COMPILER_SUPPORTS_WARNING_WEAK_VTABLES)
if(COMPILER_SUPPORTS_WARNING_WEAK_VTABLES)
  target_compile_options(AIIRSparseTensorRuntime PRIVATE
    "-Wweak-vtables")
endif()
