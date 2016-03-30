// Run cc1as asm output path just to make sure it works
// REQUIRES: x86-registered-target
// RUN: %clang -cc1as -triple x86_64-apple-macosx10.10.0 -filetype asm %s -o /dev/null
// Test that cc1as is able to consume a quoted clang full version
// RUN: %clang -cc1as -triple x86_64-apple-macosx10.10.0 -dwarf-debug-producer 'clang dummy version' -filetype asm %s -o /dev/null
