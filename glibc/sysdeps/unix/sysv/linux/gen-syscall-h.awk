# Generate SYS_* macros from a list in a text file.
# Copyright (C) 2017-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

# Emit a conditional definition for SYS_NAME.
function emit(name) {
    print "#ifdef __NR_" name;
    print "# define SYS_" name " __NR_" name;
    print "#endif";
    print "";
}

# Bail out with an error.
function fatal(message) {
    print FILENAME ":" FNR ": " message > "/dev/stderr";
    exit 1;
}

BEGIN {
    name = "";
    kernel = "";
}

# Skip empty lines and comments.
/^\s*(|#.*)$/ {
    next;
}

# Kernel version.  Used for documentation purposes only.
/^kernel [0-9.]+$/ {
    if (kernel != "") {
        fatal("duplicate kernel directive");
    }
    kernel = $2;
    print "/* Generated at libc build time from syscall list.  */";
    print "/* The system call list corresponds to kernel " kernel ".  */";
    print "";
    print "#ifndef _SYSCALL_H"
    print "# error \"Never use <bits/syscall.h> directly; include <sys/syscall.h> instead.\"";
    print "#endif";
    print "";
    split($2, kernel_version, ".");
    kernel_major = kernel_version[1];
    kernel_minor = kernel_version[2];
    kernel_version_code = kernel_major * 65536 + kernel_minor * 256;
    print "#define __GLIBC_LINUX_VERSION_CODE " kernel_version_code;
    print "";
    next;
}

# If there is just one word, it is a system call.
/^[a-zA-Z_][a-zA-Z0-9_]+$/ {
    if (kernel == "") {
        fatal("expected kernel directive before this line");
    }
    if ($1 <= name) {
        fatal("name " name " violates ordering");
    }
    emit($1);
    name = $1;
    next;
}

# The rest has to be syntax errors.
// {
    fatal("unrecognized syntax");
}
