enum {
  VALUE = 3
};

extern int glob_x;

int f(int x) {
  return x+glob_x+VALUE; 
}

typedef struct {
  int x;
  int y;
} Vector;

int vector_get_x(Vector v) {
  int x = v.x;
  return x;
}

int f(int);
int f(int);

// RUN: c-index-test \

// RUN:  -file-refs-at=%s:2:5 \
// CHECK:      EnumConstantDecl=VALUE:2:3 (Definition)
// CHECK-NEXT: EnumConstantDecl=VALUE:2:3 (Definition) =[2:3 - 2:8]
// CHECK-NEXT: DeclRefExpr=VALUE:2:3 =[8:19 - 8:24]

// RUN:  -file-refs-at=%s:8:15 \
// CHECK-NEXT: DeclRefExpr=glob_x:5:12
// CHECK-NEXT: VarDecl=glob_x:5:12 =[5:12 - 5:18]
// CHECK-NEXT: DeclRefExpr=glob_x:5:12 =[8:12 - 8:18]

// RUN:  -file-refs-at=%s:8:10 \
// CHECK-NEXT: DeclRefExpr=x:7:11
// CHECK-NEXT: ParmDecl=x:7:11 (Definition) =[7:11 - 7:12]
// CHECK-NEXT: DeclRefExpr=x:7:11 =[8:10 - 8:11]

// RUN:  -file-refs-at=%s:12:7 \
// CHECK-NEXT: FieldDecl=x:12:7 (Definition)
// CHECK-NEXT: FieldDecl=x:12:7 (Definition) =[12:7 - 12:8]
// CHECK-NEXT: MemberRefExpr=x:12:7 {{.*}} =[17:13 - 17:14]

// RUN:  -file-refs-at=%s:16:21 \
// CHECK-NEXT: TypeRef=Vector:14:3
// CHECK-NEXT: TypedefDecl=Vector:14:3 (Definition) =[14:3 - 14:9]
// CHECK-NEXT: TypeRef=Vector:14:3 =[16:18 - 16:24]

// RUN:  -file-refs-at=%s:21:5 \
// CHECK-NEXT: FunctionDecl=f:21:5
// CHECK-NEXT: FunctionDecl=f:7:5 (Definition) =[7:5 - 7:6]
// CHECK-NEXT: FunctionDecl=f:21:5 =[21:5 - 21:6]
// CHECK-NEXT: FunctionDecl=f:22:5 =[22:5 - 22:6]

// RUN:   %s | FileCheck %s
