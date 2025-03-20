// Test is line- and column-sensitive. Run lines are below

template <typename T>
class basic_vector {
public:
    T x;
    T y;
};

using my_vec = basic_vector<int>;

class MyClass {
    my_vec myVec;
};

struct OuterStruct {
    struct InnerStruct;
    int outer_field;
};

// RUN: c-index-test -single-symbol-sgf-at=%s:13:13 local %s | FileCheck --check-prefix=CHECK-MYVEC %s
// CHECK-MYVEC: "parentContexts":[{"kind":"c++.class","name":"MyClass","usr":"c:@S@MyClass"},{"kind":"c++.property","name":"myVec","usr":"c:@S@MyClass@FI@myVec"}]
// CHECK-MYVEC: "identifier":{"interfaceLanguage":"c++","precise":"c:@S@MyClass@FI@myVec"}
// CHECK-MYVEC: "kind":{"displayName":"Instance Property","identifier":"c++.property"}
// CHECK-MYVEC: "title":"myVec"
// CHECK-MYVEC: "pathComponents":["MyClass","myVec"]

// RUN: c-index-test -single-symbol-sgf-at=%s:17:17 local %s | FileCheck --check-prefix=CHECK-INNER %s
// CHECK-INNER: "parentContexts":[{"kind":"c++.struct","name":"OuterStruct","usr":"c:@S@OuterStruct"},{"kind":"c++.struct","name":"InnerStruct","usr":"c:@S@OuterStruct@S@InnerStruct"}]
// CHECK-INNER: "identifier":{"interfaceLanguage":"c++","precise":"c:@S@OuterStruct@S@InnerStruct"}
// CHECK-INNER: "kind":{"displayName":"Structure","identifier":"c++.struct"}
// CHECK-INNER: "title":"InnerStruct"
// CHECK-INNER: "pathComponents":["OuterStruct","InnerStruct"]
