// RUN: %clang_cc1 -triple i686--windows-msvc -fsyntax-only %s
// RUN: not %clang_cc1 -triple i686--windows-msvc -fsyntax-only -fdwarf-exceptions %s 2>&1 | FileCheck -check-prefix=MSVC-X86-DWARF %s
// RUN: not %clang_cc1 -triple i686--windows-msvc -fsyntax-only -fseh-exceptions %s 2>&1 | FileCheck -check-prefix=MSVC-X86-SEH %s
// RUN: not %clang_cc1 -triple i686--windows-msvc -fsyntax-only -fsjlj-exceptions %s 2>&1 | FileCheck -check-prefix=MSVC-X86-SJLJ %s

// RUN: %clang_cc1 -triple x86_64--windows-msvc -fsyntax-only %s
// RUN: not %clang_cc1 -triple x86_64--windows-msvc -fsyntax-only -fdwarf-exceptions %s 2>&1 | FileCheck -check-prefix=MSVC-X64-DWARF %s
// RUN: not %clang_cc1 -triple x86_64--windows-msvc -fsyntax-only -fseh-exceptions %s 2>&1 | FileCheck -check-prefix=MSVC-X64-SEH %s
// RUN: not %clang_cc1 -triple x86_64--windows-msvc -fsyntax-only -fsjlj-exceptions %s 2>&1 | FileCheck -check-prefix=MSVC-X64-SJLJ %s

// RUN: %clang_cc1 -triple i686--windows-gnu -fsyntax-only %s
// RUN: %clang_cc1 -triple i686--windows-gnu -fsyntax-only -fdwarf-exceptions %s
// RUN: %clang_cc1 -triple i686--windows-gnu -fsyntax-only -fseh-exceptions %s
// RUN: %clang_cc1 -triple i686--windows-gnu -fsyntax-only -fsjlj-exceptions %s

// RUN: %clang_cc1 -triple x86_64--windows-gnu -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64--windows-gnu -fsyntax-only -fdwarf-exceptions %s
// RUN: %clang_cc1 -triple x86_64--windows-gnu -fsyntax-only -fseh-exceptions %s
// RUN: %clang_cc1 -triple x86_64--windows-gnu -fsyntax-only -fsjlj-exceptions %s

// MSVC-X86-DWARF: error: invalid exception model 'fdwarf-exceptions' for target 'i686-unknown-windows-msvc'
// MSVC-X86-SEH: error: invalid exception model 'fseh-exceptions' for target 'i686-unknown-windows-msvc'
// MSVC-X86-SJLJ: error: invalid exception model 'fsjlj-exceptions' for target 'i686-unknown-windows-msvc'

// MSVC-X64-DWARF: error: invalid exception model 'fdwarf-exceptions' for target 'x86_64-unknown-windows-msvc'
// MSVC-X64-SEH: error: invalid exception model 'fseh-exceptions' for target 'x86_64-unknown-windows-msvc'
// MSVC-X64-SJLJ: error: invalid exception model 'fsjlj-exceptions' for target 'x86_64-unknown-windows-msvc'
