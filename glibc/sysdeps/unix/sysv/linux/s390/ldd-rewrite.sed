/LD_TRACE_LOADED_OBJECTS=1/a\
add_env="$add_env LD_LIBRARY_VERSION=\\$verify_out"

# ldd is generated from elf/ldd.bash.in with the name
# of ld.so as generated in Makeconfig

# that name is replaced by a pair referring to both
# the 32bit and 64bit dynamic linker.

# /lib(64|)/*(64|).so* is replaced with /lib/*.so* and /lib/*64.so*
# this works for /lib64/ld64.so.x and /lib/ld.so.x as input
s_lib64_lib_
s_64\.so_\.so_
s_^RTLDLIST=\(.*lib\)\(/[^/]*\)\(\.so\.[0-9.]*\)[[:blank:]]*$_RTLDLIST="\1\2\3 \1\264\3"_

