template <typename T>
struct Base {
    void func();
    int operator[](T);
};

struct Sub: public Base<int> {
    void func();
};

template <typename T>
inline T myMax(T a, T b)
{ return (a > b) ? a : b; }

int main()
{
    Sub inst;
    inst.func();
    inst[1];
    inst.operator[](1);

    inst.Base<int>::operator[](1);
    myMax<int>(1, 2);

    return 0;
}

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: cursor-ref-names.cpp:17:5: DeclStmt= Extent=[17:5 - 17:14]
// CHECK: cursor-ref-names.cpp:17:9: VarDecl=inst:17:9 (Definition) Extent=[17:5 - 17:13]
// CHECK: cursor-ref-names.cpp:17:5: TypeRef=struct Sub:7:8 Extent=[17:5 - 17:8]
// CHECK: cursor-ref-names.cpp:17:9: CallExpr=Sub:7:8 Extent=[17:9 - 17:13]
// CHECK: cursor-ref-names.cpp:18:5: CallExpr=func:8:10 Extent=[18:5 - 18:16]
// CHECK: cursor-ref-names.cpp:18:10: MemberRefExpr=func:8:10 SingleRefName=[18:10 - 18:14] RefName=[18:10 - 18:14] Extent=[18:5 - 18:14]
// CHECK: cursor-ref-names.cpp:18:5: DeclRefExpr=inst:17:9 Extent=[18:5 - 18:9]
// CHECK: cursor-ref-names.cpp:19:5: CallExpr=operator[]:4:9 Extent=[19:5 - 19:12] 
// CHECK: cursor-ref-names.cpp:19:5: DeclRefExpr=inst:17:9 Extent=[19:5 - 19:9]
// CHECK: cursor-ref-names.cpp:19:9: DeclRefExpr=operator[]:4:9 Extent=[19:9 - 19:10]
// CHECK: cursor-ref-names.cpp:20:5: CallExpr=operator[]:4:9 Extent=[20:5 - 20:23]
// CHECK: cursor-ref-names.cpp:20:10: MemberRefExpr=operator[]:4:9 SingleRefName=[20:10 - 20:20] RefName=[20:10 - 20:18] RefName=[20:18 - 20:19] RefName=[20:19 - 20:20] Extent=[20:5 - 20:20]
// CHECK: cursor-ref-names.cpp:20:5: DeclRefExpr=inst:17:9 Extent=[20:5 - 20:9]
// CHECK: cursor-ref-names.cpp:22:5: CallExpr=operator[]:4:9 Extent=[22:5 - 22:34]
// CHECK: cursor-ref-names.cpp:22:21: MemberRefExpr=operator[]:4:9 SingleRefName=[22:10 - 22:31] RefName=[22:10 - 22:21] RefName=[22:21 - 22:29] RefName=[22:29 - 22:30] RefName=[22:30 - 22:31] Extent=[22:5 - 22:31]
// CHECK: cursor-ref-names.cpp:22:5: DeclRefExpr=inst:17:9 Extent=[22:5 - 22:9]
// CHECK: cursor-ref-names.cpp:22:10: TemplateRef=Base:2:8 Extent=[22:10 - 22:14]
// CHECK: cursor-ref-names.cpp:23:5: CallExpr=myMax:12:10 Extent=[23:5 - 23:21]
// CHECK: cursor-ref-names.cpp:23:5: DeclRefExpr=myMax:12:10 RefName=[23:5 - 23:10] RefName=[23:10 - 23:15] Extent=[23:5 - 23:15]
