  # AOCC selective LLVM source-based coverage (link flag)
  # Injected by llvm_instrumentation.sh for --patch_coverage on upstream branches
  string(LENGTH "${LLVM_COVERAGE_FILES}" _lcf_len)
  if (${_lcf_len} GREATER 0)
    set_property(TARGET ${name} APPEND_STRING PROPERTY
    LINK_FLAGS " -fprofile-instr-generate")
  endif()
