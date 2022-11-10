! Verify that the plugin passed to -fpass-plugin is loaded and run

! UNSUPPORTED: system-windows

! REQUIRES: plugins, shell, examples

! RUN: %flang -S %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s
! RUN: %flang_fc1 -S %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s

! CHECK: Running pass: {{.*}}Bye on empty_

subroutine empty
end subroutine empty
