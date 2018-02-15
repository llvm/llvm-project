#!/usr/bin/awk -f

#===-- generate_netbsd_syscalls.awk ----------------------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
#
# This file is a generator of:
#  - include/sanitizer/netbsd_syscall_hooks.h
#  - lib/sanitizer_common/sanitizer_syscalls_netbsd.inc
#
# This script accepts on the input syscalls.master by default located in the
# /usr/src/sys/kern/syscalls.master path in the NetBSD distribution.
#
# NetBSD version 8.0.
#
#===------------------------------------------------------------------------===#

BEGIN {
  # harcode the script name
  script_name = "generate_netbsd_syscalls.awk"
  outputh = "../include/sanitizer/netbsd_syscall_hooks.h"
  outputinc = "../lib/sanitizer_common/sanitizer_syscalls_netbsd.inc"

  # assert that we are in the directory with scripts
  in_utils = system("test -f " script_name " && exit 1 || exit 0")
  if (in_utils == 0) {
    usage()
  }

  # assert 1 argument passed
  if (ARGC != 2) {
    usage()
  }

  # assert argument is a valid file path to syscall.master
  if (system("test -f " ARGV[1]) != 0) {
    usage()
  }

  # sanity check that the path ends with "syscall.master"
  if (ARGV[1] !~ /syscalls\.master$/) {
    usage()
  }

  # accept overloading CLANGFORMAT from environment
  clangformat = "clang-format"
  if ("CLANGFORMAT" in ENVIRON) {
    clangformat = ENVIRON["CLANGFORMAT"]
  }

  # parsing specific symbols
  parsingheader=1

  parsedsyscalls=0

  # Hardcoded in algorithm
  SYS_MAXSYSARGS=8
}

# Parse the RCS ID from syscall.master
parsingheader == 1 && NR == 1 {
  if (match($0, /\$[^$]+\$/)) {
    # trim initial 'NetBSD: ' and trailing ' $'
    syscallmasterversion = substr($0, RSTART + 9, RLENGTH - 11)
  } else {
    # wrong file?
    usage()
  }
}

# skip the following lines
#  - empty
NF == 0 {
  next
}
#  - comment
$1 == ";" {
  next
}

# separator between the header and table with syscalls
$0 == "%%" {
  parsingheader = 0
  next
}

# preserve 'if/elif/else/endif' C preprocessor as-is
parsingheader == 0 && $0 ~ /^#/ {
  if (parsedsyscalls in ifelifelseendif) {
    ifelifelseendif[parsedsyscalls] = ifelifelseendif[parsedsyscalls] "\n" $0
  } else {
    ifelifelseendif[parsedsyscalls] = $0
  }
  next
}

# parsing of syscall definitions
parsingheader == 0 && $1 ~ /^[0-9]+$/ {
  # first join multiple lines into single one
  while (sub(/\\$/, "")) {
    getline line
    $0 = $0 "" line
  }

  # Skip unwanted syscalls
  skip=0
  if ($0 ~ /OBSOL/ || $0 ~ /EXCL/ || $0 ~ /UNIMPL/) {
    skip=1
  }

  # Compose the syscall name
  #  - compat?
  compat=""
  if (match($0, /COMPAT_[0-9]+/)) {
    compat = tolower(substr($0, RSTART, RLENGTH))
  }
  # - alias name?
  alias=""
  if ($(NF) != "}" && !skip) {
    alias = alias "" $(NF)
  }
  # - compat version?
  compatver=""
  if (match($0, /\|[0-9]+\|/)) {
    compatver = tolower(substr($0, RSTART + 1, RLENGTH - 2))
  }
  # - basename?
  basename=""
  if (skip) {
    basename = $1
  } else {
    if (match($0, /\|[_a-z0-9]+\(/)) {
      basename = tolower(substr($0, RSTART + 1, RLENGTH - 2))
    }
  }

  syscallname=""

  if (skip) {
    syscallname= syscallname "$"
  }

  if (length(compat) > 0) {
    syscallname = syscallname "" compat "_"
  }
  if (length(alias) > 0) {
    syscallname = syscallname "" alias
  } else {
    if (length(compatver) > 0) {
      syscallname = syscallname "__" basename "" compatver
    } else {
      syscallname = syscallname "" basename
    }
  }

  # Store the syscallname
  syscalls[parsedsyscalls]=syscallname

  # Extract syscall arguments
  if (match($0, /\([^)]+\)/)) {
    args = substr($0, RSTART + 1, RLENGTH - 2)

    if (args == "void") {
      syscallargs[parsedsyscalls] = "void"
      syscallfullargs[parsedsyscalls] = "void"
    } else {
      # Normalize 'type * argument' to 'type *argument'
      gsub("\\*[ \t]+", "*", args)

      n = split(args, a, ",")

      # Handle the first argument
      match(a[1], /[*_a-z0-9\[\]]+$/)
      syscallfullargs[parsedsyscalls] = substr(a[1], RSTART) "_"

      gsub(".+[ *]", "", a[1])
      syscallargs[parsedsyscalls] = a[1]

      # Handle the rest of arguments
      for (i = 2; i <= n; i++) {
        match(a[i], /[*_a-zA-Z0-9\[\]]+$/)
        fs = substr(a[i], RSTART)
        if (fs ~ /\[/) {
          sub(/\[/, "_[", fs)
        } else {
          fs = fs "_"
        }
        syscallfullargs[parsedsyscalls] = syscallfullargs[parsedsyscalls] "$" fs
	gsub(".+[ *]", "", a[i])
        syscallargs[parsedsyscalls] = syscallargs[parsedsyscalls] "$" a[i]
      }

      # Handle array arguments for syscall(2) and __syscall(2)
      nargs = "arg0$arg1$arg2$arg3$arg4$arg5$arg6$arg7"
      gsub(/args\[SYS_MAXSYSARGS\]/, nargs, syscallargs[parsedsyscalls])
    }
  }

  parsedsyscalls++

  # Done with this line
  next
}


END {
  # empty file?
  if (NR < 1 && !abnormal_exit) {
    usage()
  }

  # Handle abnormal exit
  if (abnormal_exit) {
    exit(abnormal_exit)
  }

  # Generate sanitizer_syscalls_netbsd.inc

  # open pipe
  cmd = clangformat " > " outputh

  pcmd("//===-- netbsd_syscall_hooks.h --------------------------------------------===//")
  pcmd("//")
  pcmd("//                     The LLVM Compiler Infrastructure")
  pcmd("//")
  pcmd("// This file is distributed under the University of Illinois Open Source")
  pcmd("// License. See LICENSE.TXT for details.")
  pcmd("//")
  pcmd("//===----------------------------------------------------------------------===//")
  pcmd("//")
  pcmd("// This file is a part of public sanitizer interface.")
  pcmd("//")
  pcmd("// System call handlers.")
  pcmd("//")
  pcmd("// Interface methods declared in this header implement pre- and post- syscall")
  pcmd("// actions for the active sanitizer.")
  pcmd("// Usage:")
  pcmd("//   __sanitizer_syscall_pre_getfoo(...args...);")
  pcmd("//   long long res = syscall(SYS_getfoo, ...args...);")
  pcmd("//   __sanitizer_syscall_post_getfoo(res, ...args...);")
  pcmd("//")
  pcmd("// DO NOT EDIT! THIS FILE HAS BEEN GENERATED!")
  pcmd("//")
  pcmd("// Generated with: " script_name)
  pcmd("// Generated date: " strftime("%F"))
  pcmd("// Generated from: " syscallmasterversion)
  pcmd("//")
  pcmd("//===----------------------------------------------------------------------===//")
  pcmd("#ifndef SANITIZER_NETBSD_SYSCALL_HOOKS_H")
  pcmd("#define SANITIZER_NETBSD_SYSCALL_HOOKS_H")
  pcmd("")

  # TODO

  pcmd("")
  pcmd("#ifdef __cplusplus")
  pcmd("extern \"C\" {")
  pcmd("#endif")
  pcmd("")
  pcmd("// Private declarations. Do not call directly from user code. Use macros above.")
  pcmd("")
  pcmd("// DO NOT EDIT! THIS FILE HAS BEEN GENERATED!")
  pcmd("")

  # TODO

  pcmd("")
  pcmd("#ifdef __cplusplus")
  pcmd("} // extern \"C\"")
  pcmd("#endif")

  pcmd("")
  pcmd("// DO NOT EDIT! THIS FILE HAS BEEN GENERATED!")
  pcmd("")

  pcmd("#endif  // SANITIZER_NETBSD_SYSCALL_HOOKS_H")

  close(cmd)

  # Generate sanitizer_syscalls_netbsd.inc

  # open pipe
  cmd = clangformat " > " outputinc

  pcmd("//===-- sanitizer_syscalls_netbsd.inc ---------------------------*- C++ -*-===//")
  pcmd("//")
  pcmd("//                     The LLVM Compiler Infrastructure")
  pcmd("//")
  pcmd("// This file is distributed under the University of Illinois Open Source")
  pcmd("// License. See LICENSE.TXT for details.")
  pcmd("//")
  pcmd("//===----------------------------------------------------------------------===//")
  pcmd("//")
  pcmd("// Common syscalls handlers for tools like AddressSanitizer,")
  pcmd("// ThreadSanitizer, MemorySanitizer, etc.")
  pcmd("//")
  pcmd("// This file should be included into the tool's interceptor file,")
  pcmd("// which has to define it's own macros:")
  pcmd("//   COMMON_SYSCALL_PRE_READ_RANGE")
  pcmd("//          Called in prehook for regions that will be read by the kernel and")
  pcmd("//          must be initialized.")
  pcmd("//   COMMON_SYSCALL_PRE_WRITE_RANGE")
  pcmd("//          Called in prehook for regions that will be written to by the kernel")
  pcmd("//          and must be addressable. The actual write range may be smaller than")
  pcmd("//          reported in the prehook. See POST_WRITE_RANGE.")
  pcmd("//   COMMON_SYSCALL_POST_READ_RANGE")
  pcmd("//          Called in posthook for regions that were read by the kernel. Does")
  pcmd("//          not make much sense.")
  pcmd("//   COMMON_SYSCALL_POST_WRITE_RANGE")
  pcmd("//          Called in posthook for regions that were written to by the kernel")
  pcmd("//          and are now initialized.")
  pcmd("//   COMMON_SYSCALL_ACQUIRE(addr)")
  pcmd("//          Acquire memory visibility from addr.")
  pcmd("//   COMMON_SYSCALL_RELEASE(addr)")
  pcmd("//          Release memory visibility to addr.")
  pcmd("//   COMMON_SYSCALL_FD_CLOSE(fd)")
  pcmd("//          Called before closing file descriptor fd.")
  pcmd("//   COMMON_SYSCALL_FD_ACQUIRE(fd)")
  pcmd("//          Acquire memory visibility from fd.")
  pcmd("//   COMMON_SYSCALL_FD_RELEASE(fd)")
  pcmd("//          Release memory visibility to fd.")
  pcmd("//   COMMON_SYSCALL_PRE_FORK()")
  pcmd("//          Called before fork syscall.")
  pcmd("//   COMMON_SYSCALL_POST_FORK(long long res)")
  pcmd("//          Called after fork syscall.")
  pcmd("//")
  pcmd("// DO NOT EDIT! THIS FILE HAS BEEN GENERATED!")
  pcmd("//")
  pcmd("// Generated with: " script_name)
  pcmd("// Generated date: " strftime("%F"))
  pcmd("// Generated from: " syscallmasterversion)
  pcmd("//")
  pcmd("//===----------------------------------------------------------------------===//")
  pcmd("")
  pcmd("#include \"sanitizer_platform.h\"")
  pcmd("#if SANITIZER_NETBSD")
  pcmd("")
  pcmd("#include \"sanitizer_libc.h\"")
  pcmd("")
  pcmd("#define PRE_SYSCALL(name)                                                      \\")
  pcmd("  SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_syscall_pre_impl_##name")
  pcmd("#define PRE_READ(p, s) COMMON_SYSCALL_PRE_READ_RANGE(p, s)")
  pcmd("#define PRE_WRITE(p, s) COMMON_SYSCALL_PRE_WRITE_RANGE(p, s)")
  pcmd("")
  pcmd("#define POST_SYSCALL(name)                                                     \\")
  pcmd("  SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_syscall_post_impl_##name")
  pcmd("#define POST_READ(p, s) COMMON_SYSCALL_POST_READ_RANGE(p, s)")
  pcmd("#define POST_WRITE(p, s) COMMON_SYSCALL_POST_WRITE_RANGE(p, s)")
  pcmd("")
  pcmd("#ifndef COMMON_SYSCALL_ACQUIRE")
  pcmd("# define COMMON_SYSCALL_ACQUIRE(addr) ((void)(addr))")
  pcmd("#endif")
  pcmd("")
  pcmd("#ifndef COMMON_SYSCALL_RELEASE")
  pcmd("# define COMMON_SYSCALL_RELEASE(addr) ((void)(addr))")
  pcmd("#endif")
  pcmd("")
  pcmd("#ifndef COMMON_SYSCALL_FD_CLOSE")
  pcmd("# define COMMON_SYSCALL_FD_CLOSE(fd) ((void)(fd))")
  pcmd("#endif")
  pcmd("")
  pcmd("#ifndef COMMON_SYSCALL_FD_ACQUIRE")
  pcmd("# define COMMON_SYSCALL_FD_ACQUIRE(fd) ((void)(fd))")
  pcmd("#endif")
  pcmd("")
  pcmd("#ifndef COMMON_SYSCALL_FD_RELEASE")
  pcmd("# define COMMON_SYSCALL_FD_RELEASE(fd) ((void)(fd))")
  pcmd("#endif")
  pcmd("")
  pcmd("#ifndef COMMON_SYSCALL_PRE_FORK")
  pcmd("# define COMMON_SYSCALL_PRE_FORK() {}")
  pcmd("#endif")
  pcmd("")
  pcmd("#ifndef COMMON_SYSCALL_POST_FORK")
  pcmd("# define COMMON_SYSCALL_POST_FORK(res) {}")
  pcmd("#endif")
  pcmd("")
  pcmd("// FIXME: do some kind of PRE_READ for all syscall arguments (int(s) and such).")
  pcmd("")
  pcmd("extern \"C\" {")

  # TODO

  pcmd("}  // extern \"C\"")
  pcmd("")
  pcmd("#undef PRE_SYSCALL")
  pcmd("#undef PRE_READ")
  pcmd("#undef PRE_WRITE")
  pcmd("#undef POST_SYSCALL")
  pcmd("#undef POST_READ")
  pcmd("#undef POST_WRITE")
  pcmd("")
  pcmd("#endif  // SANITIZER_NETBSD")

  close(cmd)

  # Hack for preprocessed code
  system("sed -i 's,^ \\([^ ]\\),  \\1,' " outputinc)
}

function usage()
{
  print "Usage: " script_name " syscalls.master"
  abnormal_exit = 1
  exit 1
}

function pcmd(string)
{
  print string | cmd
}
