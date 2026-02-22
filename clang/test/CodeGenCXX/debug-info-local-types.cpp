// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -O0 -emit-llvm \
// RUN:   -disable-llvm-passes -debug-info-kind=limited | FileCheck %s
//
// Test that types declared inside functions, that receive an "identifier"
// field used for ODR-uniquing, are placed inside the declaration DISubprogram
// for the function rather than the definition DISubprogram. This avoids
// later problems with distinct types in distinct DISubprograms being
// inadvertantly unique'd; see github PR 75385.
//
// NB: The types below are marked distinct, but other portions of LLVM
// force-unique them at a later date, see the enableDebugTypeODRUniquing
// feature. Clang doesn't enable that itself; instead this test ensures a safe
// representation of the types is produced.
//
// The check-lines below are not strictly in order of hierachy, so here's a
// diagram of what's desired:
//
//                  DIFile
//                    |
//          Decl-DISubprogram "foo"
//          /                      \
//         /                        \
// Def-DISubprogram "foo"    DICompositeType "bar"
//                                   |
//                                   |
//                          Decl-DISubprogram "get_a"
//                         /         |
//                        /          |
// Def-DISubprogram "get_a"    DICompositeType "baz"
//                                   |
//                                   |
//                        {Def,Decl}-DISubprogram "get_b"

// CHECK: ![[FILENUM:[0-9]+]] = !DIFile(filename: "{{.*}}debug-info-local-types.cpp",

// CHECK: ![[BARSTRUCT:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "bar", scope: ![[FOOFUNC:[0-9]+]], file: ![[FILENUM]],
// CHECK-SAME: identifier: "_ZTSZ3foovE3bar")

// CHECK: ![[FOOFUNC]] = !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: ![[FILENUM]], file: ![[FILENUM]],
//// Test to ensure that this is _not_ a definition, therefore a decl.
// CHECK-SAME: spFlags: 0)

// CHECK: ![[GETADECL:[0-9]+]] = !DISubprogram(name: "get_a", scope: ![[BARSTRUCT]], file: ![[FILENUM]],
//// Test to ensure that this is _not_ a definition, therefore a decl.
// CHECK-SAME: spFlags: 0)

// CHECK: ![[BAZSTRUCT:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "baz", scope: ![[GETADECL]], file: ![[FILENUM]],
// CHECK-SAME: identifier: "_ZTSZZ3foovEN3bar5get_aEvE3baz")
// CHECK: distinct !DISubprogram(name: "get_b",
// CHECK-SAME: scope: ![[BAZSTRUCT]], file: ![[FILENUM]],

inline int foo() {
  class bar {
  private:
    int a = 0;
  public:
    int get_a() {
      class baz {
      private:
        int b = 0;
      public:
        int get_b() {
          return b;
        }
      };

      static baz xyzzy;
      return a + xyzzy.get_b();
    }
  };

  static bar baz;
  return baz.get_a();
}

int a() {
  return foo();
}

