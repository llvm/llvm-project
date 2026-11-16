// A link.exe-style host linker has no GNU '-l' or '-L' options. Dash-prefixed
// MSVC linker options that start with 'l' or 'L' must be forwarded verbatim
// to the host linker rather than being misparsed as '-l <lib>' or '-L <dir>'.

// The wrapper only scans the input for offload sections, which '--dry-run'
// skips, so an empty file stands in for a real object.
// RUN: touch %t.o

// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link %t.o -link -LTCG -LARGEADDRESSAWARE \
// RUN:   /link -libpath:%t.dir -out:%t.exe 2>&1 | FileCheck %s --check-prefix=MSVC

// MSVC-NOT: error
// MSVC: "/usr/bin/lld-link" {{.*}}.o -link -LTCG -LARGEADDRESSAWARE /link -libpath:{{.*}} -out:{{.*}}.exe

// GNU-style host linkers still treat '-l' and '-L' as library options.
// RUN: not clang-linker-wrapper --host-triple=x86_64-w64-windows-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld %t.o -L%t.dir -link -o %t.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GNU

// GNU: error: unable to find library -link
