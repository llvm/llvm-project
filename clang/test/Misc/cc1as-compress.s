// REQUIRES: zlib
// REQUIRES: x86-registered-target

// RUN: %clang -cc1as -triple i686 --compress-debug-sections %s -o /dev/null
// RUN: %clang -cc1as -triple i686 -compress-debug-sections=zlib %s -o /dev/null
// RUN: %if zstd %{ %clang -cc1as -triple x86_64 -compress-debug-sections=zstd %s -o /dev/null %}
