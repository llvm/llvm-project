// CHECK: [[@LINE+1]]:12 | class/ObjC | Base | [[Base_USR:.*]] | {{.*}} | Decl | rel: 0
@interface Base
@end

// CHECK: [[@LINE+1]]:11 | protocol/ObjC | Prot1 | [[Prot1_USR:.*]] | {{.*}} | Decl | rel: 0
@protocol Prot1
@end

// CHECK: [[@LINE+3]]:11 | protocol/ObjC | Prot2 | [[Prot2_USR:.*]] | {{.*}} | Decl | rel: 0
// CHECK: [[@LINE+2]]:17 | protocol/ObjC | Prot1 | [[Prot1_USR]] | {{.*}} | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | Prot2 | [[Prot2_USR]]
@protocol Prot2<Prot1>
@end

// CHECK: [[@LINE+7]]:12 | class/ObjC | Sub | [[Sub_USR:.*]] | {{.*}} | Decl | rel: 0
// CHECK: [[@LINE+6]]:18 | class/ObjC | Base | [[Base_USR]] | {{.*}} | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | Sub | [[Sub_USR]]
// CHECK: [[@LINE+4]]:23 | protocol/ObjC | Prot2 | [[Prot2_USR]] | {{.*}} | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | Sub | [[Sub_USR]]
// CHECK: [[@LINE+2]]:30 | protocol/ObjC | Prot1 | [[Prot1_USR]] | {{.*}} | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | Sub | [[Sub_USR]]
@interface Sub : Base<Prot2, Prot1>
// CHECK-NOT: [[@LINE+1]]:3 | class/ObjC | Sub |
-(Sub*)getit;
@end

// CHECK: [[@LINE+1]]:7 | class/C++ | Cls | [[Cls_USR:.*]] | {{.*}} | Def | rel: 0
class Cls {};

// CHECK: [[@LINE+3]]:7 | class/C++ | SubCls1 | [[SubCls1_USR:.*]] | {{.*}} | Def | rel: 0
// CHECK: [[@LINE+2]]:24 | class/C++ | Cls | [[Cls_USR]] | {{.*}} | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | SubCls1 | [[SubCls1_USR]]
class SubCls1 : public Cls {
  // CHECK-NOT: [[@LINE+1]]:3 | class/C++ | SubCls1 |
  SubCls1 *f;
};
