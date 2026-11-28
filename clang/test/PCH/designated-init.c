// Test this without pch.
// RUN: %clang_cc1 -source-date-epoch 1613676318 %s -include %s.h -emit-llvm -o %t.withoutpch.ll

// Test with pch.
// RUN: %clang_cc1 %s.h -emit-pch -o %t.pch
// RUN: %clang_cc1 -source-date-epoch 1613676318 %s -include-pch %t.pch -emit-llvm -o %t.withpch.ll
// RUN: diff %t.withoutpch.ll %t.withpch.ll
