// clang-format off
// REQUIRES: lld

// Test various interesting cases for AST reconstruction.
// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/ast-types.lldbinit 2>&1 | FileCheck %s

// Test trivial versions of each tag type.
class TrivialC {};
struct TrivialS {};
union TrivialU {};
enum TrivialE {TE_A};

// Test reconstruction of DeclContext hierarchies.
namespace A {
  namespace B {
    template<typename T>
    struct C {
      T ABCMember;
    };

    // Let's try a template specialization with a different implementation
    template<>
    struct C<void> {
      void *ABCSpecializationMember;
    };
  }

  // Let's make sure we can distinguish classes and namespaces.  Also let's try
  // a non-type template parameter.
  template<int N>
  struct C {
    class D {
      int ACDMember = 0;
      C<N - 1> *CPtr = nullptr;
    };
  };

  struct D {
    // Let's make a nested class with the same name as another nested class
    // elsewhere, and confirm that they appear in the right DeclContexts in
    // the AST.
    struct E {
      int ADDMember;
    };
  };
}


// Let's try an anonymous namespace.
namespace {
  template<typename T>
  struct Anonymous {
    int AnonymousMember;
    // And a nested class within an anonymous namespace
    struct D {
      int AnonymousDMember;
    };
  };
}

TrivialC TC;
TrivialS TS;
TrivialU TU;
TrivialE TE;

A::B::C<int> ABCInt;
A::B::C<float> ABCFloat;
A::B::C<void> ABCVoid;

A::C<0> AC0;
A::C<-1> ACNeg1;

// FIXME: The type `D` is located now at the level of the translation unit.
// FIXME: Should be located in the namespace `A`, in the struct `C<1>`.
A::C<1>::D AC1D;

A::C<0>::D AC0D;
A::C<-1>::D ACNeg1D;
A::D AD;
A::D::E ADE;

Anonymous<int> AnonInt;
Anonymous<A::B::C<void>> AnonABCVoid;
Anonymous<A::B::C<void>>::D AnonABCVoidD;

// FIXME: Enum size isn't being correctly determined.
// FIXME: Can't read memory for variable values.

// CHECK: (TrivialC) TC = {}
// CHECK: (TrivialS) TS = {}
// CHECK: (TrivialU) TU = {}
// CHECK: (TrivialE) TE = TE_A
// CHECK: (A::B::C<int>) ABCInt = (ABCMember = 0)
// CHECK: (A::B::C<float>) ABCFloat = (ABCMember = 0)
// CHECK: (A::B::C<void>) ABCVoid = (ABCSpecializationMember = 0x{{0+}})
// CHECK: (A::C<0>) AC0 = {}
// CHECK: (A::C<-1>) ACNeg1 = {}
// CHECK: (A::C<1>::D) AC1D = (ACDMember = 0, CPtr = 0x{{0+}})
// CHECK: (A::C<0>::D) AC0D = (ACDMember = 0, CPtr = 0x{{0+}})
// CHECK: (A::C<-1>::D) ACNeg1D = (ACDMember = 0, CPtr = 0x{{0+}})
// CHECK: (A::D) AD = {}
// CHECK: (A::D::E) ADE = (ADDMember = 0)
// CHECK: ((anonymous namespace)::Anonymous<int>) AnonInt = (AnonymousMember = 0)
// CHECK: ((anonymous namespace)::Anonymous<A::B::C<void>>) AnonABCVoid = (AnonymousMember = 0)
// CHECK: ((anonymous namespace)::Anonymous<A::B::C<void>>::D) AnonABCVoidD = (AnonymousDMember = 0)
// CHECK: Dumping clang ast for 1 modules.
// CHECK: TranslationUnitDecl {{.*}}
// CHECK: |-CXXRecordDecl {{.*}} class TrivialC definition
// CHECK: |-CXXRecordDecl {{.*}} struct TrivialS definition
// CHECK: |-CXXRecordDecl {{.*}} union TrivialU definition
// CHECK: |-EnumDecl {{.*}} TrivialE
// CHECK: | `-EnumConstantDecl {{.*}} TE_A 'TrivialE'
// CHECK: |-NamespaceDecl {{.*}} A
// CHECK: | |-NamespaceDecl {{.*}} B
// CHECK: | | |-CXXRecordDecl {{.*}} struct C<int> definition
// CHECK: | | | `-FieldDecl {{.*}} ABCMember 'int'
// CHECK: | | |-CXXRecordDecl {{.*}} struct C<float> definition
// CHECK: | | | `-FieldDecl {{.*}} ABCMember 'float'
// CHECK: | | `-CXXRecordDecl {{.*}} struct C<void> definition
// CHECK: | |   `-FieldDecl {{.*}} ABCSpecializationMember 'void *'
// FIXME: | |-CXXRecordDecl {{.*}} struct C<1> definition
// FIXME: | | `-CXXRecordDecl {{.*}} class D definition
// FIXME: | |   |-FieldDecl {{.*}} ACDMember 'int'
// FIXME: | |   `-FieldDecl {{.*}} CPtr 'A::C<1> *'
// CHECK: | |-CXXRecordDecl {{.*}} struct C<0> definition
// CHECK: | | `-CXXRecordDecl {{.*}} class D definition
// CHECK: | |   |-FieldDecl {{.*}} ACDMember 'int'
// CHECK: | |   `-FieldDecl {{.*}} CPtr 'A::C<-1> *'
// CHECK: | |-CXXRecordDecl {{.*}} struct C<-1> definition
// CHECK: | | `-CXXRecordDecl {{.*}} class D definition
// CHECK: | |   |-FieldDecl {{.*}} ACDMember 'int'
// CHECK: | |   `-FieldDecl {{.*}} CPtr 'A::C<-2> *'
// CHECK: | |-CXXRecordDecl {{.*}} struct C<-2>
// CHECK: | `-CXXRecordDecl {{.*}} struct D definition
// CHECK: |   `-CXXRecordDecl {{.*}} struct E definition
// CHECK: |     `-FieldDecl {{.*}} ADDMember 'int'
// CHECK: |-NamespaceDecl
// CHECK: | |-CXXRecordDecl {{.*}} struct Anonymous<int> definition
// CHECK: | | `-FieldDecl {{.*}} AnonymousMember 'int'
// CHECK: | `-CXXRecordDecl {{.*}} struct Anonymous<A::B::C<void>> definition
// CHECK: |   |-FieldDecl {{.*}} AnonymousMember 'int'
// CHECK: |   `-CXXRecordDecl {{.*}} struct D definition
// CHECK: |     `-FieldDecl {{.*}} AnonymousDMember 'int'

int main(int argc, char **argv) {
  AnonInt.AnonymousMember = 1;
  AnonABCVoid.AnonymousMember = 2;
  AnonABCVoidD.AnonymousDMember = 3;

  return 0;
}
