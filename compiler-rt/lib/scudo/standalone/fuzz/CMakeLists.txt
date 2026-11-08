if (LLVM_USE_SANITIZE_COVERAGE)
  add_executable(allocator_fuzzer
      allocator_fuzzer.cpp)
  set_target_properties(
      allocator_fuzzer PROPERTIES FOLDER "Fuzzers")
  target_compile_options(
      allocator_fuzzer PRIVATE -fsanitize=fuzzer)
  set_target_properties(
      allocator_fuzzer PROPERTIES LINK_FLAGS -fsanitize=fuzzer)
  target_include_directories(
      allocator_fuzzer PRIVATE .. ../include)
endif()
