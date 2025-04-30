/Maybe extra code for non-ELF binaries/a\
  file=$1\
  # Run the ldd stub.\
  lddlibc4 "$file"\
  # Test the result.\
  if test $? -lt 3; then\
    return 0;\
  fi\
  # In case of an error punt.
/LD_TRACE_LOADED_OBJECTS=1/a\
add_env="$add_env LD_LIBRARY_VERSION=\\$verify_out"
