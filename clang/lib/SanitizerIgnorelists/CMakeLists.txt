set(INSTRUMENTATION_RESOURCE_FILES
    asan_ignorelist.txt
    hwasan_ignorelist.txt
    msan_ignorelist.txt
    cfi_ignorelist.txt
)

get_clang_resource_dir(resource_output_dir PREFIX ${LLVM_LIBRARY_OUTPUT_INTDIR}/.. SUBDIR share)
get_clang_resource_dir(resource_install_dir SUBDIR share)

set(sanitizer_ignorelist_outputs)

foreach(file_name ${INSTRUMENTATION_RESOURCE_FILES})
    set(src_file "${CMAKE_CURRENT_SOURCE_DIR}/${file_name}")
    set(dst_file "${resource_output_dir}/${file_name}")

    add_custom_command(OUTPUT ${dst_file}
        DEPENDS ${src_file}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_file} ${dst_file}
        COMMENT "Copying ${file_name}...")
    list(APPEND sanitizer_ignorelist_outputs ${dst_file})
endforeach()

add_custom_target(sanitizer-ignorelists DEPENDS ${sanitizer_ignorelist_outputs})
set_target_properties(sanitizer-ignorelists PROPERTIES FOLDER "Clang/Resources")

install(
FILES ${INSTRUMENTATION_RESOURCE_FILES}
DESTINATION ${resource_install_dir}
COMPONENT sanitizer-ignorelists)

if(NOT LLVM_ENABLE_IDE)
add_llvm_install_targets("install-sanitizer-ignorelists"
                        DEPENDS sanitizer-ignorelists
                        COMPONENT sanitizer-ignorelists)
endif()