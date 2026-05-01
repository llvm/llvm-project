// REQUIRES: x86-registered-target
// RUN: %clang_cl --target=i686-windows-msvc /c /Z7 /Fo%t.obj -- %s
// RUN: llvm-pdbutil dump --types %t.obj | FileCheck %s
// RUN: %clang_cl -gcodeview-command-line --target=i686-windows-msvc /c /Z7 /Fo%t.obj -- %s
// RUN: llvm-pdbutil dump --types %t.obj | FileCheck %s
// RUN: %clang_cl --target=i686-windows-msvc /c /Z7 /Fo%t.obj -fdebug-compilation-dir=. -- %s
// RUN: llvm-pdbutil dump --types %t.obj | FileCheck %s --check-prefix RELATIVE
// RUN: %clang_cl -gno-codeview-command-line --target=i686-windows-msvc /c /Z7 /Fo%t.obj -- %s
// RUN: llvm-pdbutil dump --types %t.obj | FileCheck %s --check-prefix DISABLE

// -fmessage-length shouldn't be included in the command line since it breaks reproducibility
// RUN: %clang_cl -gcodeview-command-line --target=i686-windows-msvc -Xclang -fmessage-length=100 /c /Z7 /Fo%t.obj -- %s
// RUN: llvm-pdbutil dump --types %t.obj | FileCheck %s --check-prefix MESSAGELEN

// The source filename must be stripped from the embedded cc1 command line
// whether it's passed to the driver as an absolute or a relative path.
// See https://github.com/llvm/llvm-project/issues/193900.

// RUN: %clang_cl --target=i686-windows-msvc /c /Z7 /Fo%t.obj -- %s
// RUN: llvm-pdbutil dump --types %t.obj | FileCheck %s --check-prefix ABSPATH

// RUN: rm -rf %t.relpath && mkdir %t.relpath
// RUN: cp %s %t.relpath/hello.cpp
// RUN: cd %t.relpath && %clang_cl --target=i686-windows-msvc /c /Z7 /Fo:hello.obj -- hello.cpp
// RUN: llvm-pdbutil dump --types %t.relpath/hello.obj | FileCheck %s --check-prefix RELPATH

int main(void) { return 42; }

// CHECK:                       Types (.debug$T)
// CHECK: ============================================================
// CHECK: 0x[[PWD:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String: [[PWDVAL:.+]]
// CHECK: 0x[[FILEPATH:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String: [[FILEPATHVAL:.+[\\/]codeview-buildinfo.c]]
// CHECK: 0x[[ZIPDB:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String:
// CHECK: 0x[[TOOL:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String: [[TOOLVAL:.+[\\/][clang|llvm].*]]
// CHECK: 0x[[CMDLINE:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String: "-cc1
// CHECK: 0x{{.+}} | LF_BUILDINFO [size = {{.+}}]
// CHECK-NEXT:          0x[[PWD]]: `[[PWDVAL]]`
// CHECK-NEXT:          0x[[TOOL]]: `[[TOOLVAL]]`
// CHECK-NEXT:          0x[[FILEPATH]]: `[[FILEPATHVAL]]`
// CHECK-NEXT:          0x[[ZIPDB]]: ``
// CHECK-NEXT:          0x[[CMDLINE]]: `"-cc1

// RELATIVE:                       Types (.debug$T)
// RELATIVE: ============================================================
// RELATIVE: 0x{{.+}} | LF_BUILDINFO [size = {{.+}}]
// RELATIVE:          0x{{.+}}: `.`

// DISABLE-NOT: "-cc1"
// DISABLE: 0x{{.+}} | LF_BUILDINFO [size = {{.+}}]
// DISABLE-NEXT:          0x{{.+}}: `{{.*}}`
// DISABLE-NEXT:          0x{{.+}}: `{{.*}}`
// DISABLE-NEXT:          0x{{.+}}: `{{.*}}`
// DISABLE-NEXT:          0x{{.+}}: ``
// DISABLE-NEXT:          0x{{.+}}: `{{.*}}`

// MESSAGELEN:                       Types (.debug$T)
// MESSAGELEN: ============================================================
// MESSAGELEN: 0x{{.+}} | LF_BUILDINFO [size = {{.+}}]
// MESSAGELEN-NOT: -fmessage-length

// The cmdline is the 5th argument of LF_BUILDINFO. The source filename must
// not appear inside its value (the SourceFile field, the 3rd argument, is
// reserved for that).
// ABSPATH:       0x{{.+}} | LF_BUILDINFO [size = {{.+}}]
// ABSPATH-NEXT:           0x{{.*}}: `{{.*}}`
// ABSPATH-NEXT:           0x{{.*}}: `{{.*}}`
// ABSPATH-NEXT:           0x{{.*}}: `{{.+[\\/]codeview-buildinfo\.c}}`
// ABSPATH-NEXT:           0x{{.*}}: ``
// ABSPATH-NEXT:           0x{{.*}}: `
// ABSPATH-NOT:   {{[^"]*[\\/]codeview-buildinfo\.c}}
// ABSPATH-SAME:  `

// RELPATH:       0x{{.+}} | LF_BUILDINFO [size = {{.+}}]
// RELPATH-NEXT:           0x{{.*}}: `{{.*}}`
// RELPATH-NEXT:           0x{{.*}}: `{{.*}}`
// RELPATH-NEXT:           0x{{.*}}: `{{.*hello\.cpp}}`
// RELPATH-NEXT:           0x{{.*}}: ``
// RELPATH-NEXT:           0x{{.*}}: `
// RELPATH-NOT:   {{hello\.cpp}}
// RELPATH-SAME:  `
