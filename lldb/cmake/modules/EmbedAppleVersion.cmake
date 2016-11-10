execute_process(COMMAND /usr/libexec/PlistBuddy -c "Print:CFBundleVersion" ${LLDB_INFO_PLIST}
                OUTPUT_VARIABLE BundleVersion
                OUTPUT_STRIP_TRAILING_WHITESPACE)

file(APPEND "${HEADER_FILE}.tmp"
    "#define LLDB_VERSION_STRING lldb-${BundleVersion}\n")

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${HEADER_FILE}.tmp" "${HEADER_FILE}")

file(REMOVE "${HEADER_FILE}.tmp")
