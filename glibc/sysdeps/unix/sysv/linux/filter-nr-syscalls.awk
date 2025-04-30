# Filter preprocessor __NR_* macros and extract system call names.
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

# Skip reserved system calls.
/^#define __NR_(unused|reserved)[0-9]+ / {
    next;
}

# Skip pseudo-system calls which describe ranges.
/^#define __NR_(syscalls|arch_specific_syscall|(OABI_)?SYSCALL_BASE) / {
    next;
}
/^#define __NR_(|64_|[NO]32_)Linux(_syscalls)? / {
    next;
}

# Print the remaining _NR_* macros as system call names.
/^#define __NR_/ {
    print substr($2, 6);
}
