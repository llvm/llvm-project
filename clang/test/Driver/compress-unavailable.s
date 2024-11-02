// RUN: %clang -### -fintegrated-as -gz=none -c %s 2>&1 | FileCheck %s --check-prefix=NOWARN
// NOWARN-NOT: warning: cannot compress debug sections (zlib not enabled)

// RUN: %if !zlib %{ %clang -### -fintegrated-as -gz -c %s 2>&1 | FileCheck %s --check-prefix=WARN-ZLIB %}
// WARN-ZLIB: warning: cannot compress debug sections (zlib not enabled)

// RUN: %if !zstd %{ %clang -### -fintegrated-as -gz=zstd -c %s 2>&1 | FileCheck %s --check-prefix=WARN-ZSTD %}
// WARN-ZSTD: warning: cannot compress debug sections (zstd not enabled)
