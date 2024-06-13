include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_tool(clang-doc
  ClangDocMain.cpp
  )

clang_target_link_libraries(clang-doc
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  clangToolingCore
  )
target_link_libraries(clang-doc
  PRIVATE
  clangDoc
  )

install(FILES ../assets/clang-doc-default-stylesheet.css
  DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
  COMPONENT clang-doc)

install(FILES ../assets/index.js
  DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
  COMPONENT clang-doc)

add_custom_target(copy-clang-doc-assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/../assets" "${CMAKE_BINARY_DIR}/share/clang"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../assets"
        COMMENT "Copying Clang-Doc Assets"
)
set_target_properties(copy-clang-doc-assets PROPERTIES FOLDER "Clang-Doc/Assets")
add_dependencies(clang-doc copy-clang-doc-assets)