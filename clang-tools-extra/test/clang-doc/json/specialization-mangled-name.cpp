// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/_ZTT7MyClassIiE.json

template<typename T> class MyClass;

template<> class MyClass<int>;

// CHECK:       "Name": "MyClass",
// CHECK:       "Template": {
// CHECK-NEXT:    "Specialization": {
// CHECK-NEXT:      "Parameters": [
// CHECK-NEXT:        "int"
// CHECK-NEXT:      ],
// CHECK-NEXT:      "SpecializationOf": "{{[0-9A-F]*}}"
