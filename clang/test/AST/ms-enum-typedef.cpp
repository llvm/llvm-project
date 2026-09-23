// Test without serialization:
// RUN: %clang_cc1 -std=c++11 -fms-compatibility \
// RUN:   -Wno-microsoft-enum-typedef -ast-dump -ast-dump-filter Use %s \
// RUN:   | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++11 -fms-compatibility \
// RUN:   -Wno-microsoft-enum-typedef -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++11 -fms-compatibility \
// RUN:   -Wno-microsoft-enum-typedef -include-pch %t -ast-dump-all \
// RUN:   -ast-dump-filter Use /dev/null \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck %s

typedef enum Underlying { Value } Alias;

struct Use {
  enum Alias *member;
};

namespace Qualified {
typedef enum Underlying { Value } Alias;
}

struct QualifiedUse {
  enum Qualified::Alias *member;
};

template <class T>
struct DependentUse {
  enum T::Alias *member;
};

// CHECK: CXXRecordDecl {{.*}} struct Use definition
// CHECK: FieldDecl {{.*}} member 'enum Alias *'
// CHECK: CXXRecordDecl {{.*}} struct QualifiedUse definition
// CHECK: FieldDecl {{.*}} member 'enum Qualified::Alias *'
// CHECK: CXXRecordDecl {{.*}} struct DependentUse definition
// CHECK: FieldDecl {{.*}} member 'enum T::Alias *'
