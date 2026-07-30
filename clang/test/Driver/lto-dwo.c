// Confirm that -gsplit-dwarf=DIR is passed to linker

// DEFINE: %{RUN-ELF} = %clang --target=x86_64-unknown-linux -### %s \
// DEFINE:              -flto=thin -gsplit-dwarf

// RUN: %{RUN-ELF} -o a.out 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-ELF-DWO-DIR-DEFAULT < %t %s
// RUN: %clang_cl --target=x86_64-unknown-windows-msvc -### -fuse-ld=lld -flto -gsplit-dwarf -o a.out -- %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-COFF-DWO-DIR-DEFAULT < %t %s
//
// CHECK-LINK-ELF-DWO-DIR-DEFAULT:  "-plugin-opt=dwo_dir=a.out-dwo"
// CHECK-LINK-COFF-DWO-DIR-DEFAULT: "/dwodir:a.out_dwo"

// Check -dumpdir effect on -gsplit-dwarf.
//
// DEFINE: %{RUN-DUMPDIR} = %{RUN-ELF} -dumpdir /dir/file.ext
//
// RUN: %{RUN-ELF} 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O
// RUN: %{RUN-ELF} -o FOO 2>&1 | FileCheck %s -check-prefix=CHECK-O
// RUN: %{RUN-DUMPDIR} 2>&1 | FileCheck %s -check-prefix=CHECK-DUMPDIR
// RUN: %{RUN-DUMPDIR} -o FOO 2>&1 | FileCheck %s -check-prefix=CHECK-DUMPDIR
//
//    CHECK-NO-O: "-plugin-opt=dwo_dir=a-dwo"
//       CHECK-O: "-plugin-opt=dwo_dir=FOO-dwo"
// CHECK-DUMPDIR: "-plugin-opt=dwo_dir=/dir/file.extdwo"
