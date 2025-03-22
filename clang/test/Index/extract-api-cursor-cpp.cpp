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

// RUN: c-index-test -single-symbol-sgf-at=%s:13:7 local %s | FileCheck --check-prefix=CHECK-VEC-TYPE %s
// CHECK-VEC-TYPE: "parentContexts":[{"kind":"c++.typealias","name":"my_vec","usr":"c:@my_vec"}]
// CHECK-VEC-TYPE: "declarationFragments":[{"kind":"keyword","spelling":"typedef"},{"kind":"text","spelling":" "},{"kind":"typeIdentifier","preciseIdentifier":"c:@ST>1#T@basic_vector","spelling":"basic_vector"},{"kind":"text","spelling":"<"},{"kind":"typeIdentifier","preciseIdentifier":"c:I","spelling":"int"},{"kind":"text","spelling":"> "},{"kind":"identifier","spelling":"my_vec"},{"kind":"text","spelling":";"}]
// CHECK-VEC-TYPE: "identifier":{"interfaceLanguage":"c++","precise":"c:@my_vec"}
// CHECK-VEC-TYPE: "kind":{"displayName":"Type Alias","identifier":"c++.typealias"}
// CHECK-VEC-TYPE: "title":"my_vec"
// CHECK-VEC-TYPE: "pathComponents":["my_vec"]

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
