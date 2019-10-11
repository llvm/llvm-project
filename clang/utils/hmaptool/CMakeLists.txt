set(CLANG_HMAPTOOL hmaptool)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/${CLANG_HMAPTOOL}
                   COMMAND ${CMAKE_COMMAND} -E make_directory
                     ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin
                   COMMAND ${CMAKE_COMMAND} -E copy
                     ${CMAKE_CURRENT_SOURCE_DIR}/${CLANG_HMAPTOOL}
                     ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CLANG_HMAPTOOL})

list(APPEND Depends ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/${CLANG_HMAPTOOL})
install(PROGRAMS ${CLANG_HMAPTOOL}
        DESTINATION bin
        COMPONENT hmaptool)

add_custom_target(hmaptool ALL DEPENDS ${Depends})
set_target_properties(hmaptool PROPERTIES FOLDER "Utils")

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-hmaptool"
                           DEPENDS hmaptool
                           COMPONENT hmaptool)
endif()
