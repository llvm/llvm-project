
add_custom_target(libcxx-generate-public-header-transitive-inclusion-tests
    COMMAND "${Python3_EXECUTABLE}" "${LIBCXX_SOURCE_DIR}/utils/generate_header_inclusion_tests.py"
    COMMENT "Generate tests checking for mandated transitive includes in public headers.")

add_custom_target(libcxx-generate-public-header-tests
    COMMAND "${Python3_EXECUTABLE}" "${LIBCXX_SOURCE_DIR}/utils/generate_header_tests.py"
    COMMENT "Generate tests for including public headers.")

add_custom_target(libcxx-generate-feature-test-macros
    COMMAND "${Python3_EXECUTABLE}" "${LIBCXX_SOURCE_DIR}/utils/generate_feature_test_macro_components.py"
    COMMENT "Generate the <version> header and tests for feature test macros.")

add_custom_target(libcxx-generate-extended-grapheme-cluster-tables
    COMMAND
        "${Python3_EXECUTABLE}"
        "${LIBCXX_SOURCE_DIR}/utils/generate_extended_grapheme_cluster_table.py"
        "${LIBCXX_SOURCE_DIR}/include/__format/extended_grapheme_cluster_table.h"
    COMMENT "Generate the extended grapheme cluster header.")

add_custom_target(libcxx-generate-extended-grapheme-cluster-tests
    COMMAND
        "${Python3_EXECUTABLE}"
        "${LIBCXX_SOURCE_DIR}/utils/generate_extended_grapheme_cluster_test.py"
         "${LIBCXX_SOURCE_DIR}/test/libcxx/utilities/format/format.string/format.string.std/extended_grapheme_cluster.h"
    COMMENT "Generate the extended grapheme cluster header.")

add_custom_target(libcxx-generate-escaped-output-table
    COMMAND
        "${Python3_EXECUTABLE}"
        "${LIBCXX_SOURCE_DIR}/utils/generate_escaped_output_table.py"
        "${LIBCXX_SOURCE_DIR}/include/__format/escaped_output_table.h"
    COMMENT "Generate the escaped output header")

add_custom_target(libcxx-generate-iwyu-mapping
    COMMAND
        "${Python3_EXECUTABLE}"
        "${LIBCXX_SOURCE_DIR}/utils/generate_iwyu_mapping.py"
    COMMENT "Generate the mapping file for include-what-you-use")

add_custom_target(libcxx-generate-files
    DEPENDS libcxx-generate-public-header-transitive-inclusion-tests
            libcxx-generate-public-header-tests
            libcxx-generate-feature-test-macros
            libcxx-generate-extended-grapheme-cluster-tables
            libcxx-generate-extended-grapheme-cluster-tests
            libcxx-generate-escaped-output-table
            libcxx-generate-iwyu-mapping
    COMMENT "Create all the auto-generated files in libc++ and its tests.")
