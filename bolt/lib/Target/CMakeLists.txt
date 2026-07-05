foreach (tgt ${BOLT_TARGETS_TO_BUILD})
  add_subdirectory(${tgt})
endforeach()
