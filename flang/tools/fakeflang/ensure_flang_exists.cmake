# Copy INPUT_FILE to OUTPUT_FILE, but only
# if OUTPUT_FILE does not already exist

if (NOT EXISTS "${OUTPUT_FILE}")
  cmake_path(GET OUTPUT_FILE PARENT_PATH OUTPUT_DIR)
  file(MAKE_DIRECTORY "${OUTPUT_DIR}")

  # This could also be the symlink but its small size is not worth the effort
  # handling platform differences.
  file(COPY_FILE "${INPUT_FILE}" "${OUTPUT_FILE}")
endif ()
