// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/_ZTV7MyClass.json --check-prefix=BASE
// RUN: FileCheck %s < %t/json/_ZTV7MyClassIiE.json --check-prefix=SPECIALIZATION

template<typename T> struct MyClass {};

template<> struct MyClass<int> {};

// BASE:       "MangledName": "_ZTV7MyClass",
// BASE-NEXT:  "Name": "MyClass",
// BASE-NEXT:  "Namespace": [
// BASE-NEXT:    "GlobalNamespace"
// BASE-NEXT:  ],
// BASE-NEXT:  "Path": "GlobalNamespace",
// BASE-NEXT:  "TagType": "struct",
// BASE-NEXT:  "Template": {
// BASE-NEXT:    "Parameters": [
// BASE-NEXT:      "typename T"
// BASE-NEXT:    ]
// BASE-NEXT:  },

// SPECIALIZATION:       "MangledName": "_ZTV7MyClassIiE",
// SPECIALIZATION-NEXT:  "Name": "MyClass",
// SPECIALIZATION-NEXT:  "Namespace": [
// SPECIALIZATION-NEXT:    "GlobalNamespace"
// SPECIALIZATION-NEXT:  ],
// SPECIALIZATION-NEXT:  "Path": "GlobalNamespace",
// SPECIALIZATION-NEXT:  "TagType": "struct",
// SPECIALIZATION-NEXT:  "Template": {
// SPECIALIZATION-NEXT:    "Specialization": {
// SPECIALIZATION-NEXT:      "Parameters": [
// SPECIALIZATION-NEXT:        "int"
// SPECIALIZATION-NEXT:      ],
// SPECIALIZATION-NEXT:      "SpecializationOf": "{{[0-9A-F]*}}"
// SPECIALIZATION-NEXT:    }
// SPECIALIZATION-NEXT:  },
