/LD_TRACE_LOADED_OBJECTS=1/a\
add_env="$add_env LD_LIBRARY_VERSION=\\$verify_out"
s_^\(RTLDLIST=\)\(.*lib\)\(\|64\)\(/[^/]*\.so\.[0-9.]*\)[ 	]*$_\1"\2\4 \264\4"_
