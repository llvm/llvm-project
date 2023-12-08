int *a;
int * _Nonnull b;
int * _Nullable c;
int * _Null_unspecified d;
int * _Nullable_result e;

// RUN: env CINDEXTEST_INCLUDE_ATTRIBUTED_TYPES=1 c-index-test -test-print-type %s | FileCheck %s
// CHECK: VarDecl=a:1:6 [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: VarDecl=b:2:16 [type=int * _Nonnull] [typekind=Attributed] [nullability=nonnull] [canonicaltype=int *] [canonicaltypekind=Pointer] [modifiedtype=int *] [modifiedtypekind=Pointer] [isPOD=1]
// CHECK: VarDecl=c:3:17 [type=int * _Nullable] [typekind=Attributed] [nullability=nullable] [canonicaltype=int *] [canonicaltypekind=Pointer] [modifiedtype=int *] [modifiedtypekind=Pointer] [isPOD=1]
// CHECK: VarDecl=d:4:25 [type=int * _Null_unspecified] [typekind=Attributed] [nullability=unspecified] [canonicaltype=int *] [canonicaltypekind=Pointer] [modifiedtype=int *] [modifiedtypekind=Pointer] [isPOD=1]
// CHECK: VarDecl=e:5:24 [type=int * _Nullable_result] [typekind=Attributed] [nullability=nullable_result] [canonicaltype=int *] [canonicaltypekind=Pointer] [modifiedtype=int *] [modifiedtypekind=Pointer] [isPOD=1]
