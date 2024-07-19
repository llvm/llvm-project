// Test that --print-supported-extensions errors on unsupported architectures.

// RUN: %if x86-registered-target %{ not %clang --target=x86_64-linux-gnu \
// RUN:   --print-supported-extensions 2>&1 | FileCheck %s --check-prefix X86 %}
// X86: error: option '--print-supported-extensions' cannot be specified on this target
