// Test the mtime validation behavior of a relocatable PCH when source files
// are relocated and their timestamps change.  Mtime drift is expected after
// relocation, so a bare mtime mismatch is tolerated.  With
// -fvalidate-ast-input-files-content an mtime mismatch falls back to a
// content hash comparison, which still catches genuine content changes.

// RUN: rm -rf %t
// RUN: mkdir -p %t/sysroot_orig/usr/include

// Create a simple header in the original sysroot.
// RUN: echo '#ifndef TEST_H'            > %t/sysroot_orig/usr/include/test.h
// RUN: echo '#define TEST_H'           >> %t/sysroot_orig/usr/include/test.h
// RUN: echo 'int relocated_val = 42;'  >> %t/sysroot_orig/usr/include/test.h
// RUN: echo '#endif'                   >> %t/sysroot_orig/usr/include/test.h

// Generate a relocatable PCH against the original sysroot.  Include
// -fvalidate-ast-input-files-content so that content hashes are stored in
// the PCH and can be compared on subsequent loads.
// RUN: %clang_cc1 -x c-header -relocatable-pch -emit-pch \
// RUN:   -isysroot %t/sysroot_orig -fvalidate-ast-input-files-content \
// RUN:   -o %t/test.pch %t/sysroot_orig/usr/include/test.h

// Baseline: loading with the original sysroot succeeds.
// RUN: %clang_cc1 -include-pch %t/test.pch \
// RUN:   -isysroot %t/sysroot_orig \
// RUN:   -fsyntax-only %s

// Set up a new sysroot by moving the header there (simulating workspace relocation).
// RUN: mkdir -p %t/sysroot_new/usr/include
// RUN: mv %t/sysroot_orig/usr/include/test.h %t/sysroot_new/usr/include/test.h

// Loading the relocatable PCH with the new sysroot and with mtime preserved works.
// RUN: %clang_cc1 -include-pch %t/test.pch \
// RUN:   -isysroot %t/sysroot_new \
// RUN:   -fsyntax-only %s

// Advance mtime of the relocated header by 1 hour to simulate timestamp drift
// that naturally occurs when files are relocated, without changing content.
// RUN: %python -c "import os,sys; t=os.path.getmtime(sys.argv[1])+3600; os.utime(sys.argv[1],(t,t))" \
// RUN:   %t/sysroot_new/usr/include/test.h

// Mtime changed but content is identical: the relocatable PCH is accepted.
// RUN: %clang_cc1 -include-pch %t/test.pch \
// RUN:   -isysroot %t/sysroot_new \
// RUN:   -fsyntax-only %s

// With content validation the mtime mismatch falls back to a content hash
// comparison; since the content is unchanged the PCH loads successfully.
// RUN: %clang_cc1 -include-pch %t/test.pch \
// RUN:   -isysroot %t/sysroot_new -fvalidate-ast-input-files-content \
// RUN:   -fsyntax-only %s

// Overwrite the header with different content of the same size (42 -> 69).
// RUN: echo '#ifndef TEST_H'            > %t/sysroot_new/usr/include/test.h
// RUN: echo '#define TEST_H'           >> %t/sysroot_new/usr/include/test.h
// RUN: echo 'int relocated_val = 69;'  >> %t/sysroot_new/usr/include/test.h
// RUN: echo '#endif'                   >> %t/sysroot_new/usr/include/test.h

// Advance the mtime again to ensure it differs from the value stored in the PCH.
// On a fast machine, the newly written file might have the same mtime as the original.
// RUN: %python -c "import os,sys; t=os.path.getmtime(sys.argv[1])+3600; os.utime(sys.argv[1],(t,t))" \
// RUN:   %t/sysroot_new/usr/include/test.h

// Without content validation, mtime drift is tolerated and the content change
// goes undetected.
// RUN: %clang_cc1 -include-pch %t/test.pch \
// RUN:   -isysroot %t/sysroot_new \
// RUN:   -fsyntax-only %s

// With content validation the change is caught even though size is the same.
// RUN: not %clang_cc1 -include-pch %t/test.pch \
// RUN:   -isysroot %t/sysroot_new -fvalidate-ast-input-files-content \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK-CONTENT,CHECK

// CHECK: fatal error: file '{{.*}}test.h' has been modified since the precompiled header '{{.*}}test.pch' was built
// CHECK-CONTENT: note: content changed

int get_val(void) { return relocated_val; }
