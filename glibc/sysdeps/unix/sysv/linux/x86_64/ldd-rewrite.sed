/LD_TRACE_LOADED_OBJECTS=1/a\
add_env="$add_env LD_LIBRARY_VERSION=\\$verify_out"
s_^\(RTLDLIST=\)\(.*lib\)\(\|64\|x32\)\(/[^/]*\)\(-x86-64\|-x32\)\(\.so\.[0-9.]*\)[ 	]*$_\1"\2\4\6 \264\4-x86-64\6 \2x32\4-x32\6"_
