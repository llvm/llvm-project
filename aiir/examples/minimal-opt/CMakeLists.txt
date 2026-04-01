set(LLVM_LINK_COMPONENTS
  Support
  )


set(LIBS
  AIIRParser
  AIIRSupport
  AIIRIR
)

add_aiir_tool(aiir-cat
  aiir-cat.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
  )
target_link_libraries(aiir-cat PRIVATE ${LIBS})
llvm_update_compile_flags(aiir-cat)
aiir_check_all_link_libraries(aiir-cat)

list(APPEND LIBS
  AIIROptLib
  AIIRPass
)
add_aiir_tool(aiir-minimal-opt
  aiir-minimal-opt.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
  )
target_link_libraries(aiir-minimal-opt PRIVATE ${LIBS})
llvm_update_compile_flags(aiir-minimal-opt)
aiir_check_all_link_libraries(aiir-minimal-opt)


list(APPEND LIBS
  AIIROptLib
  AIIRPass
  AIIRTransforms
)
add_aiir_tool(aiir-minimal-opt-canonicalize
  aiir-minimal-opt-canonicalize.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
  )
target_link_libraries(aiir-minimal-opt-canonicalize PRIVATE ${LIBS})
llvm_update_compile_flags(aiir-minimal-opt-canonicalize)
aiir_check_all_link_libraries(aiir-minimal-opt-canonicalize)

