setup_host_tool(clang-tidy-confusable-chars-gen CLANG_TIDY_CONFUSABLE_CHARS_GEN clang_tidy_confusable_chars_gen clang_tidy_confusable_chars_gen_target)

add_subdirectory(ConfusableTable)

add_custom_command(
    OUTPUT Confusables.inc
    COMMAND ${clang_tidy_confusable_chars_gen} ${CMAKE_CURRENT_SOURCE_DIR}/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
    DEPENDS ${clang_tidy_confusable_chars_gen_target} ConfusableTable/confusables.txt)

add_custom_target(genconfusable DEPENDS Confusables.inc)
set_target_properties(genconfusable PROPERTIES FOLDER "Clang Tools Extra/Sourcegenning")
