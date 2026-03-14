// RUN: %clang_cc1 -triple i686--windows-msvc -fsyntax-only %s
// RUN: not %clang_cc1 -triple i686--windows-msvc -fsyntax-only -exception-model=dwarf %s 2>&1 | FileCheck -check-prefix=MSVC-X86-DWARF %s
// RUN: not %clang_cc1 -triple i686--windows-msvc -fsyntax-only -exception-model=seh %s 2>&1 | FileCheck -check-prefix=MSVC-X86-SEH %s
// RUN: not %clang_cc1 -triple i686--windows-msvc -fsyntax-only -exception-model=sjlj %s 2>&1 | FileCheck -check-prefix=MSVC-X86-SJLJ %s

// RUN: %clang_cc1 -triple x86_64--windows-msvc -fsyntax-only %s
// RUN: not %clang_cc1 -triple x86_64--windows-msvc -fsyntax-only -exception-model=dwarf %s 2>&1 | FileCheck -check-prefix=MSVC-X64-DWARF %s
// RUN: not %clang_cc1 -triple x86_64--windows-msvc -fsyntax-only -exception-model=seh %s 2>&1 | FileCheck -check-prefix=MSVC-X64-SEH %s
// RUN: not %clang_cc1 -triple x86_64--windows-msvc -fsyntax-only -exception-model=sjlj %s 2>&1 | FileCheck -check-prefix=MSVC-X64-SJLJ %s

// RUN: %clang_cc1 -triple i686--windows-gnu -fsyntax-only %s
// RUN: %clang_cc1 -triple i686--windows-gnu -fsyntax-only -exception-model=dwarf %s
// RUN: %clang_cc1 -triple i686--windows-gnu -fsyntax-only -exception-model=seh %s
// RUN: %clang_cc1 -triple i686--windows-gnu -fsyntax-only -exception-model=sjlj %s

// RUN: %clang_cc1 -triple x86_64--windows-gnu -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64--windows-gnu -fsyntax-only -exception-model=dwarf %s
// RUN: %clang_cc1 -triple x86_64--windows-gnu -fsyntax-only -exception-model=seh %s
// RUN: %clang_cc1 -triple x86_64--windows-gnu -fsyntax-only -exception-model=sjlj %s

// MSVC-X86-DWARF: error: invalid exception model 'dwarf' for target 'i686-unknown-windows-msvc'
// MSVC-X86-SEH: error: invalid exception model 'seh' for target 'i686-unknown-windows-msvc'
// MSVC-X86-SJLJ: error: invalid exception model 'sjlj' for target 'i686-unknown-windows-msvc'

// MSVC-X64-DWARF: error: invalid exception model 'dwarf' for target 'x86_64-unknown-windows-msvc'
// MSVC-X64-SEH: error: invalid exception model 'seh' for target 'x86_64-unknown-windows-msvc'
// MSVC-X64-SJLJ: error: invalid exception model 'sjlj' for target 'x86_64-unknown-windows-msvc'
