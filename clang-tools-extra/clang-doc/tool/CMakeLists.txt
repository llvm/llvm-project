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


set(assets
  index.js
  clang-doc-default-stylesheet.css
)

set(asset_dir "${CMAKE_CURRENT_SOURCE_DIR}/../assets")
set(resource_dir "${CMAKE_BINARY_DIR}/share/clang")
set(out_files)

function(copy_files_to_dst src_dir dst_dir file)
  set(src "${src_dir}/${file}")
  set(dst "${dst_dir}/${file}")
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying ${file} to ${dst_dir}"
  )
  list(APPEND out_files ${dst})
  set(out_files ${out_files} PARENT_SCOPE)
endfunction(copy_files_to_dst)

foreach(f ${assets})
  install(FILES ${asset_dir}/${f}
    DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
    COMPONENT clang-doc)
  copy_files_to_dst(${asset_dir} ${resource_dir} ${f})
endforeach(f)

add_custom_target(copy-clang-doc-assets
  DEPENDS ${out_files}
  COMMENT "Copying Clang-Doc Assets"
)
set_target_properties(copy-clang-doc-assets PROPERTIES FOLDER "Clang-Doc/Assets")
add_dependencies(clang-doc copy-clang-doc-assets)
