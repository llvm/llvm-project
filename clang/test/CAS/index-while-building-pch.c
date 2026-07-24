// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -o %t/deps_pch.json -- \
// RUN:   %clang -x c-header %t/pch.h -o %t/pch.pch

// RUN: %deps-to-rsp %t/deps_pch.json --tu-index=0 > %t/pch.rsp
// RUN: %clang @%t/pch.rsp

// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -o %t/deps_tu.json -- \
// RUN:   %clang -c %t/tu.c -o %t/tu.o -include-pch %t/pch.pch -index-store-path %t/tu.index

// RUN: %deps-to-rsp %t/deps_tu.json --tu-index=0 > %t/tu.rsp
// FIXME: This is currently only testing that Clang doesn't crash when producing
//        an index of a TU using a PCH with compilation caching. We should
//        assert that it produces something reasonable that clients understand.
// RUN: %clang @%t/tu.rsp

//--- pch.h
//--- tu.c
