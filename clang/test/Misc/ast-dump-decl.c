// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump -ast-dump-filter Test %s | FileCheck -check-prefix CHECK -strict-whitespace %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck -check-prefix CHECK-TU -strict-whitespace %s

int TestLocation;
// CHECK: VarDecl 0x{{[^ ]*}} <{{.*}}:4:1, col:5> col:5 TestLocation

struct TestIndent {
  int x;
};
// CHECK:      {{^}}RecordDecl{{.*TestIndent[^()]*$}}
// CHECK-NEXT: {{^}}`-FieldDecl{{.*x[^()]*$}}

struct TestChildren {
  int x;
  struct y {
    int z;
  };
};
// CHECK:      RecordDecl{{.*}}TestChildren
// CHECK-NEXT:   FieldDecl{{.*}}x
// CHECK-NEXT:   RecordDecl{{.*}}y
// CHECK-NEXT:     FieldDecl{{.*}}z

// CHECK-TU: TranslationUnitDecl

void testLabelDecl() {
  __label__ TestLabelDecl;
  TestLabelDecl: goto TestLabelDecl;
}
// CHECK:      LabelDecl{{.*}} TestLabelDecl

typedef int TestTypedefDecl;
// CHECK:      TypedefDecl{{.*}} TestTypedefDecl 'int'

__module_private__ typedef int TestTypedefDeclPrivate;
// CHECK:      TypedefDecl{{.*}} TestTypedefDeclPrivate 'int' __module_private__

enum TestEnumDecl {
  testEnumDecl
};
// CHECK:      EnumDecl{{.*}} TestEnumDecl
// CHECK-NEXT:   EnumConstantDecl{{.*}} testEnumDecl

struct TestEnumDeclAnon {
  enum {
    testEnumDeclAnon
  } e;
};
// CHECK:      RecordDecl{{.*}} TestEnumDeclAnon
// CHECK-NEXT:   EnumDecl{{.*> .*$}}

enum TestEnumDeclForward;
// CHECK:      EnumDecl{{.*}} TestEnumDeclForward

__module_private__ enum TestEnumDeclPrivate;
// CHECK:      EnumDecl{{.*}} TestEnumDeclPrivate __module_private__

struct TestRecordDecl {
  int i;
};
// CHECK:      RecordDecl{{.*}} struct TestRecordDecl
// CHECK-NEXT:   FieldDecl

struct TestRecordDeclEmpty {
};
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclEmpty

struct TestRecordDeclAnon1 {
  struct {
  } testRecordDeclAnon1;
};
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclAnon1
// CHECK-NEXT:   RecordDecl{{.*}} struct

struct TestRecordDeclAnon2 {
  struct {
  };
};
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclAnon2
// CHECK-NEXT:   RecordDecl{{.*}} struct

struct TestRecordDeclForward;
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclForward

__module_private__ struct TestRecordDeclPrivate;
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclPrivate __module_private__

enum testEnumConstantDecl {
  TestEnumConstantDecl,
  TestEnumConstantDeclInit = 1
};
// CHECK:      EnumConstantDecl{{.*}} TestEnumConstantDecl 'int'
// CHECK:      EnumConstantDecl{{.*}} TestEnumConstantDeclInit 'int'
// CHECK-NEXT:   IntegerLiteral

struct testIndirectFieldDecl {
  struct {
    int TestIndirectFieldDecl;
  };
};
// CHECK:      IndirectFieldDecl{{.*}} TestIndirectFieldDecl 'int'
// CHECK-NEXT:   Field{{.*}} ''
// CHECK-NEXT:   Field{{.*}} 'TestIndirectFieldDecl'

// FIXME: It would be nice to dump the enum and its enumerators.
int TestFunctionDecl(int x, enum { e } y) {
  return x;
}
// CHECK:      FunctionDecl{{.*}} TestFunctionDecl 'int (int, enum {{.*}})'
// CHECK-NEXT:   ParmVarDecl{{.*}} x
// CHECK-NEXT:   ParmVarDecl{{.*}} y
// CHECK-NEXT:   CompoundStmt

// FIXME: It would be nice to 'Enum' and 'e'.
int TestFunctionDecl2(enum Enum { e } x) { return x; }
// CHECK:      FunctionDecl{{.*}} TestFunctionDecl2 'int (enum {{.*}})'
// CHECK-NEXT:   ParmVarDecl{{.*}} x
// CHECK-NEXT:   CompoundStmt


int TestFunctionDeclProto(int x);
// CHECK:      FunctionDecl{{.*}} TestFunctionDeclProto 'int (int)'
// CHECK-NEXT:   ParmVarDecl{{.*}} x

extern int TestFunctionDeclSC();
// CHECK:      FunctionDecl{{.*}} TestFunctionDeclSC 'int ()' extern

inline int TestFunctionDeclInline();
// CHECK:      FunctionDecl{{.*}} TestFunctionDeclInline 'int ()' inline

struct testFieldDecl {
  int TestFieldDecl;
  int TestFieldDeclWidth : 1;
  __module_private__ int TestFieldDeclPrivate;
};
// CHECK:      FieldDecl{{.*}} TestFieldDecl 'int'
// CHECK:      FieldDecl{{.*}} TestFieldDeclWidth 'int'
// CHECK-NEXT:   IntegerLiteral
// CHECK:      FieldDecl{{.*}} TestFieldDeclPrivate 'int' __module_private__

int TestVarDecl;
// CHECK:      VarDecl{{.*}} TestVarDecl 'int'

extern int TestVarDeclSC;
// CHECK:      VarDecl{{.*}} TestVarDeclSC 'int' extern

__thread int TestVarDeclThread;
// CHECK:      VarDecl{{.*}} TestVarDeclThread 'int' tls{{$}}

__module_private__ int TestVarDeclPrivate;
// CHECK:      VarDecl{{.*}} TestVarDeclPrivate 'int' __module_private__

int TestVarDeclInit = 0;
// CHECK:      VarDecl{{.*}} TestVarDeclInit 'int'
// CHECK-NEXT:   IntegerLiteral

void testParmVarDecl(int TestParmVarDecl);
// CHECK: ParmVarDecl{{.*}} TestParmVarDecl 'int'
