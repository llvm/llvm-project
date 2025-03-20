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

// RUN: c-index-test -single-symbol-sgf-at=%s:13:13 local %s | FileCheck --check-prefix=CHECK-MYVEC %s
// CHECK-MYVEC: "parentContexts":[{"kind":"c++.class","name":"MyClass","usr":"c:@S@MyClass"},{"kind":"c++.property","name":"myVec","usr":"c:@S@MyClass@FI@myVec"}]
// CHECK-MYVEC: "identifier":{"interfaceLanguage":"c++","precise":"c:@S@MyClass@FI@myVec"}
// CHECK-MYVEC: "kind":{"displayName":"Instance Property","identifier":"c++.property"}
// CHECK-MYVEC: "title":"myVec"
// CHECK-MYVEC: "pathComponents":["MyClass","myVec"]
