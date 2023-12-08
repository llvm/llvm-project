/// Don't create symlinks on Windows
// UNSUPPORTED: system-windows

/// Check the priority used when searching for tools
/// Names and locations are usually in this order:
/// <triple>-tool, tool, program path, PATH
/// (from highest to lowest priority)
/// A higher priority name found in a lower priority
/// location will win over a lower priority name in a
/// higher priority location.
/// Prefix dirs (added with -B) override the location,
/// so only name priority is accounted for, unless we fail to find
/// anything at all in the prefix.

/// Note: All matches are expected to be at the end of file paths.
/// So we match " on the end to account for build systems that
/// put the name of the compiler in the build path.
/// E.g. /build/gcc_X.Y.Z/0/...

/// Symlink clang to a new dir which will be its
/// "program path" for these tests
// RUN: rm -rf %t && mkdir -p %t
// RUN: ln -s %clang %t/clang

/// No gccs at all, nothing is found
// RUN: env "PATH=" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO_NOTREAL_GCC %s
// NO_NOTREAL_GCC-NOT: notreal-none-elf-gcc"
/// Some systems will have "gcc-x.y.z" so for this first check
/// make sure we don't find "gcc" or "gcc-x.y.z". If we do find either
/// then there is no point continuing as this copy of clang is not
/// isolated as we expected.
// NO_NOTREAL_GCC-NOT: {{/gcc[^/]*"}}

/// <triple>-gcc in program path is found
// RUN: touch %t/notreal-none-elf-gcc && chmod +x %t/notreal-none-elf-gcc
// RUN: env "PATH=" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=PROG_PATH_NOTREAL_GCC %s
// PROG_PATH_NOTREAL_GCC: notreal-none-elf-gcc"

/// <triple>-gcc on the PATH is found
// RUN: mkdir -p %t/env
// RUN: rm %t/notreal-none-elf-gcc
// RUN: touch %t/env/notreal-none-elf-gcc && chmod +x %t/env/notreal-none-elf-gcc
// RUN: env "PATH=%t/env/" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=ENV_PATH_NOTREAL_GCC %s
// ENV_PATH_NOTREAL_GCC: env/notreal-none-elf-gcc"

/// <triple>-gcc in program path is preferred to one on the PATH
// RUN: touch %t/notreal-none-elf-gcc && chmod +x %t/notreal-none-elf-gcc
// RUN: env "PATH=%t/env/" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=BOTH_NOTREAL_GCC %s
// BOTH_NOTREAL_GCC: notreal-none-elf-gcc"
// BOTH_NOTREAL_GCC-NOT: env/notreal-none-elf-gcc"

/// On program path, <triple>-gcc is preferred to plain gcc
// RUN: touch %t/gcc && chmod +x %t/gcc
// RUN: env "PATH=" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NOTREAL_GCC_PREFERRED %s
// NOTREAL_GCC_PREFERRED: notreal-none-elf-gcc"
// NOTREAL_GCC_PREFERRED-NOT: /gcc"

/// <triple>-gcc on the PATH is preferred to gcc in program path
// RUN: rm %t/notreal-none-elf-gcc
// RUN: env "PATH=%t/env/" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NOTREAL_PATH_OVER_GCC_PROG %s
// NOTREAL_PATH_OVER_GCC_PROG: env/notreal-none-elf-gcc"
// NOTREAL_PATH_OVER_GCC_PROG-NOT: /gcc"

/// <triple>-gcc on the PATH is preferred to gcc on the PATH
// RUN: rm %t/gcc
// RUN: touch %t/env/gcc && chmod +x %t/env/gcc
// RUN: env "PATH=%t/env/" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NOTREAL_PATH_OVER_GCC_PATH %s
// NOTREAL_PATH_OVER_GCC_PATH: env/notreal-none-elf-gcc"
// NOTREAL_PATH_OVER_GCC_PATH-NOT: /gcc"

/// We cannot trust clang --version, or cmake's LLVM_DEFAULT_TARGET_TRIPLE
/// to give us the one and only default triple.
/// Can't trust cmake because on Darwin, triples have a verison appended to them.
/// (and clang uses the versioned string to search)
/// Can't trust --version because it will pad 3 item triples to 4 e.g.
/// powerpc64le-linux-gnu -> powerpc64le-unknown-linux-gnu
/// (and clang uses the former to search)
/// So we write to both names which is a bit odd but still proves that the
/// lookup is working.

/// <default-triple>-gcc has lowest priority so <triple>-gcc
/// on PATH beats default triple in program path
// RUN: DEFAULT_TRIPLE=`%t/clang --version | grep "Target:" | cut -d ' ' -f2`
// RUN: touch %t/$DEFAULT_TRIPLE-gcc && chmod +x %t/$DEFAULT_TRIPLE-gcc
// RUN: touch %t/%target_triple-gcc && chmod +x %t/%target_triple-gcc
// RUN: env "PATH=%t/env/" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=DEFAULT_TRIPLE_GCC %s
// DEFAULT_TRIPLE_GCC: env/notreal-none-elf-gcc"

/// plain gcc on PATH beats default triple in program path
// RUN: rm %t/env/notreal-none-elf-gcc
// RUN: env "PATH=%t/env/" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=DEFAULT_TRIPLE_NO_NOTREAL %s
// DEFAULT_TRIPLE_NO_NOTREAL: env/gcc"
// DEFAULT_TRIPLE_NO_NOTREAL-NOT: -gcc"

/// Pick "gcc" as a fallback. Don't pick $DEFAULT_TRIPLE-gcc.
// RUN: rm %t/env/gcc
// RUN: env "PATH=%t/env/" %t/clang -### -target notreal-none-elf %s 2>&1 | \
// RUN:   FileCheck --check-prefix=DEFAULT_TRIPLE_NO_OTHERS %s
// DEFAULT_TRIPLE_NO_OTHERS: "gcc"

/// -B paths are searched separately so default triple will win
/// if put in one of those even if other paths have higher priority names
// RUN: mkdir -p %t/prefix
/// One of these will fail when $DEFAULT_TRIPLE == %target_triple
// RUN: test -f %t/$DEFAULT_TRIPLE-gcc && \
// RUN:   mv %t/$DEFAULT_TRIPLE-gcc %t/prefix || true
// RUN: test -f %t/%target_triple-gcc && \
// RUN:   mv %t/%target_triple-gcc %t/prefix || true
// RUN: touch %t/notreal-none-elf-gcc && chmod +x %t/notreal-none-elf-gcc
// RUN: touch %t/prefix/gcc && chmod +x %t/prefix/gcc
// RUN: env "PATH=" %t/clang -### -target notreal-none-elf %s -B %t/prefix 2>&1 | \
// RUN:   FileCheck --check-prefix=DEFAULT_TRIPLE_IN_PREFIX %s
// DEFAULT_TRIPLE_IN_PREFIX: prefix/gcc"
// DEFAULT_TRIPLE_IN_PREFIX-NOT: notreal-none-elf-gcc"

/// Only if there is nothing in the prefix will we search other paths
/// -f in case $DEFAULT_TRIPLE == %target_triple
// RUN: rm -f %t/prefix/$DEFAULT_TRIPLE-gcc %t/prefix/%target_triple-gcc %t/prefix/gcc
// RUN: env "PATH=" %t/clang -### -target notreal-none-elf %s -B %t/prefix 2>&1 | \
// RUN:   FileCheck --check-prefix=EMPTY_PREFIX_DIR %s
// EMPTY_PREFIX_DIR: notreal-none-elf-gcc"
