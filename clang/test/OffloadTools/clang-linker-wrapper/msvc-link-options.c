// REQUIRES: x86-registered-target

// An MSVC-style linker has no '-l' or '-L' options. Options such as '-link'
// and '-LTCG' must be passed to the linker as they are, not read as '-l ink'
// or '-L TCG'.

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

// GNU-style linkers treat '-l' and '-L' as library options.
// RUN: not clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld %t.o -L%t.dir -link -o %t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GNU

// GNU: error: unable to find library -link

// The long forms '--library' and '--library-path' are passed as they are too.
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link %t.o --library=foo --library bar \
// RUN:   --library-path=%t.dir --library-path %t.dir -out:%t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-LONG

// MSVC-LONG-NOT: error
// MSVC-LONG: "/usr/bin/lld-link" {{.*}}.o --library=foo --library bar --library-path={{.*}} --library-path {{.*}} -out:{{.*}}.exe

// GNU-style linkers treat the long forms as library options.
// RUN: not clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld %t.o --library-path=%t.dir --library=foo \
// RUN:   -o %t.exe 2>&1 | FileCheck %s --check-prefix=GNU-LONG-EQ
// RUN: not clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld %t.o --library-path %t.dir --library bar \
// RUN:   -o %t.exe 2>&1 | FileCheck %s --check-prefix=GNU-LONG-S

// GNU-LONG-EQ: error: unable to find library -lfoo
// GNU-LONG-S: error: unable to find library -lbar

// '--help' lists '-l' and '-L' only for GNU-style linkers. Without
// '--linker-path=', an MSVC target means MSVC style.
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --help \
// RUN:   | FileCheck %s --check-prefix=MSVC-HELP
// RUN: clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --help \
// RUN:   | FileCheck %s --check-prefix=GNU-HELP
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc \
// RUN:   --linker-path=/usr/bin/ld.bfd --help | FileCheck %s --check-prefix=GNU-HELP

// MSVC-HELP: OPTIONS:
// MSVC-HELP-NOT: -L <dir>
// MSVC-HELP-NOT: -l <libname>
// GNU-HELP: -L <dir>
// GNU-HELP: -l <libname>

// A GNU linker can link an MSVC target (as with '-fuse-ld=bfd'). Then '-l'
// and '-L' are library options.
// RUN: not clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/ld.bfd %t.o -L%t.dir -lfoo -o %t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-GNU-LD
// RUN: not clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=ld %t.o -L%t.dir -lfoo -o %t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-GNU-LD

// MSVC-GNU-LD: error: unable to find library -lfoo

// Device code in a library found with '-L' and '-l' is extracted.
// RUN: rm -rf %t.lib && mkdir %t.lib
// RUN: llvm-ar rcs %t.lib/libfoo.a %t.host.o
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/ld.bfd -L%t.lib --whole-archive -lfoo \
// RUN:   --no-whole-archive -o %t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-GNU-LIB

// MSVC-GNU-LIB: clang{{.*}} --target=x86_64-unknown-linux-gnu {{.*}}.o -Wl,-Bsymbolic -shared -L {{.*}}.lib -Wl,--whole-archive -l foo -Wl,--no-whole-archive{{$}}
// MSVC-GNU-LIB: "/usr/bin/ld.bfd" -L {{.*}}.lib --whole-archive -l foo --no-whole-archive -o {{.*}}.exe {{.*}}.wrapper{{.*}}.o

// MSVC linker names keep MSVC style, with any case, with or without '.exe',
// and with a version such as 'lld-link-19' (as with '-fuse-ld=lld-link-19').
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link.exe %t.o -link -LTCG -out:%t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-NAME -DLINKER=/usr/bin/lld-link.exe
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/tools/LINK.EXE %t.o -link -LTCG -out:%t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-NAME -DLINKER=/tools/LINK.EXE
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link-19 %t.o -link -LTCG -out:%t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MSVC-NAME -DLINKER=/usr/bin/lld-link-19

// MSVC-NAME-NOT: error
// MSVC-NAME: "[[LINKER]]" {{.*}}.o -link -LTCG -out:{{.*}}.exe

// Any other linker name means GNU style, so a misread '-link' is a clear error.
// RUN: not clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/tools/my-msvc-link %t.o -link -out:%t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GNU
