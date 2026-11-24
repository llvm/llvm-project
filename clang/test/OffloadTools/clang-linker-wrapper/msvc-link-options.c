// REQUIRES: x86-registered-target

// A link.exe-style host linker has no GNU '-l' or '-L' options. Dash-prefixed
// MSVC linker options that start with 'l' or 'L' must be forwarded verbatim
// to the host linker rather than being misparsed as '-l <lib>' or '-L <dir>'.

// An empty object has no embedded device code, so only the host link is run.
// RUN: touch %t.o

// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link %t.o -link -LTCG -LARGEADDRESSAWARE \
// RUN:   /link -libpath:%t.dir -out:%t.exe 2>&1 | FileCheck %s --check-prefix=MSVC

// MSVC-NOT: error
// MSVC: "/usr/bin/lld-link" {{.*}}.o -link -LTCG -LARGEADDRESSAWARE /link -libpath:{{.*}} -out:{{.*}}.exe

// The same options must not be misparsed when the wrapper re-parses them to
// link a CPU offload image, which copies library options into the device link.
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t.host.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link %t.host.o -link -LTCG -out:%t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-CPU

// MSVC-CPU: clang{{.*}} --target=x86_64-unknown-linux-gnu {{.*}} -Wl,-Bsymbolic -shared{{$}}
// MSVC-CPU: "/usr/bin/lld-link" {{.*}}.o -link -LTCG -out:{{.*}}.exe

// GNU-style host linkers still treat '-l' and '-L' as library options.
// RUN: not clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld %t.o -L%t.dir -link -o %t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GNU

// GNU: error: unable to find library -link

// The long forms of the GNU library options are also hidden for a Windows MSVC
// host and forwarded verbatim.
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link %t.o --library=foo --library bar \
// RUN:   --library-path=%t.dir --library-path %t.dir -out:%t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-LONG

// MSVC-LONG-NOT: error
// MSVC-LONG: "/usr/bin/lld-link" {{.*}}.o --library=foo --library bar --library-path={{.*}} --library-path {{.*}} -out:{{.*}}.exe

// GNU-style host linkers still treat the long forms as library options.
// RUN: not clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld %t.o --library-path=%t.dir --library=foo \
// RUN:   -o %t.exe 2>&1 | FileCheck %s --check-prefix=GNU-LONG-EQ
// RUN: not clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld %t.o --library-path %t.dir --library bar \
// RUN:   -o %t.exe 2>&1 | FileCheck %s --check-prefix=GNU-LONG-S

// GNU-LONG-EQ: error: unable to find library -lfoo
// GNU-LONG-S: error: unable to find library -lbar

// '--help' only lists the GNU library options for GNU-style host linkers.
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --help \
// RUN:   | FileCheck %s --check-prefix=MSVC-HELP
// RUN: clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --help \
// RUN:   | FileCheck %s --check-prefix=GNU-HELP

// MSVC-HELP-NOT: -L <dir>
// MSVC-HELP-NOT: -l <libname>
// GNU-HELP: -L <dir>
// GNU-HELP: -l <libname>
