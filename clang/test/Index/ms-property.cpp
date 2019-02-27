// RUN: c-index-test core -print-source-symbols -- -fms-extensions -fno-ms-compatibility %s | FileCheck %s

// CHECK: [[@LINE+1]]:8 | struct/C++ | Simple | [[Simple_USR:.*]] | <no-cgname> | Def | rel: 0
struct Simple {
  int GetX() const;
  // CHECK: [[@LINE-1]]:7 | instance-method/C++ | GetX | [[GetX_USR:.*]] | __ZNK6Simple4GetXEv | Decl,RelChild | rel: 1
  // CHECK-NEXT: RelChild | Simple | [[Simple_USR]]

  void PutX(int i);
  // CHECK: [[@LINE-1]]:8 | instance-method/C++ | PutX | [[PutX_USR:.*]] | __ZN6Simple4PutXEi | Decl,RelChild | rel: 1
  // CHECK-NEXT: RelChild | Simple | [[Simple_USR]]

  __declspec(property(get=GetX, put=PutX)) int propX;
  // CHECK: [[@LINE-1]]:48 | instance-property/C++ | propX | [[propX_USR:.*]] | <no-cgname> | Def,RelChild | rel: 1
  // CHECK-NEXT: RelChild | Simple | [[Simple_USR]]
};

// CHECK: [[@LINE+1]]:5 | function/C | test | [[test_USR:.*]] | __Z4testv | Def | rel: 0
int test() {
  Simple s;
  s.propX = 5;
  // CHECK: [[@LINE-1]]:5 | instance-property/C++ | propX | [[propX_USR]] | <no-cgname> | Ref,RelCont | rel: 1
  // CHECK-NEXT: RelCont | test | [[test_USR]]
  // CHECK: [[@LINE-3]]:5 | instance-method/C++ | PutX | [[PutX_USR]] | __ZN6Simple4PutXEi | Ref,Call,RelCall,RelCont | rel: 1
  // CHECK-NEXT: RelCall,RelCont | test | [[test_USR]]

  return s.propX;
  // CHECK: [[@LINE-1]]:12 | instance-property/C++ | propX | [[propX_USR]] | <no-cgname> | Ref,RelCont | rel: 1
  // CHECK-NEXT: RelCont | test | [[test_USR]]
  // CHECK: [[@LINE-3]]:12 | instance-method/C++ | GetX | [[GetX_USR]] | __ZNK6Simple4GetXEv | Ref,Call,RelCall,RelCont | rel: 1
  // CHECK-NEXT: RelCall,RelCont | test | [[test_USR]]
}
