  # AOCC selective LLVM source-based coverage (per-file instrumentation)
  # Injected by llvm_instrumentation.sh for --patch_coverage on upstream branches
  get_property(_lcf_sources TARGET ${name} PROPERTY SOURCES)
  foreach(_lcf_fn ${_lcf_sources})
     get_filename_component(_lcf_abs ${_lcf_fn} ABSOLUTE)
     get_filename_component(_lcf_adir ${PROJECT_SOURCE_DIR} DIRECTORY)
     string(REPLACE ${_lcf_adir}/ "" _lcf_cf ${_lcf_abs})
     if(${_lcf_cf} IN_LIST LLVM_COVERAGE_FILES)
        set_property(SOURCE ${_lcf_fn} APPEND_STRING PROPERTY
                    COMPILE_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
        set_property(TARGET ${name} APPEND_STRING PROPERTY
                    LINK_FLAGS " -fprofile-instr-generate")
     endif()
  endforeach()
