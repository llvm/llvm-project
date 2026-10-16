// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/inheritance.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV7MyClass.json

// CHECK:       "Bases": [
// CHECK-NEXT:    {
// CHECK-NEXT:     "Access": "private",
// CHECK-NEXT:      "InfoType": "record",
// CHECK-NEXT:      "IsParent": true,
// CHECK-NEXT:      "IsTypedef": false,
// CHECK-NEXT:      "IsVirtual": false,
// CHECK-NEXT:      "MangledName": "",
// CHECK-NEXT:      "Name": "Bar",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "TagType": "struct",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "Access": "private",
// CHECK-NEXT:      "InfoType": "record",
// CHECK-NEXT:      "IsParent": false,
// CHECK-NEXT:      "IsTypedef": false,
// CHECK-NEXT:      "IsVirtual": false,
// CHECK-NEXT:      "MangledName": "",
// CHECK-NEXT:      "Name": "Foo",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "TagType": "struct",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "Access": "private",
// CHECK-NEXT:      "InfoType": "record",
// CHECK-NEXT:      "IsParent": false,
// CHECK-NEXT:      "IsTypedef": false,
// CHECK-NEXT:      "IsVirtual": true,
// CHECK-NEXT:      "MangledName": "",
// CHECK-NEXT:      "Name": "Virtual",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "TagType": "struct",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "Access": "private",
// CHECK-NEXT:      "InfoType": "record",
// CHECK-NEXT:      "IsParent": true,
// CHECK-NEXT:      "IsTypedef": false,
// CHECK-NEXT:      "IsVirtual": false,
// CHECK-NEXT:      "MangledName": "",
// CHECK-NEXT:      "Name": "Buzz",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "TagType": "struct",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "Access": "private",
// CHECK-NEXT:      "InfoType": "record",
// CHECK-NEXT:      "IsParent": false,
// CHECK-NEXT:      "IsTypedef": false,
// CHECK-NEXT:      "IsVirtual": false,
// CHECK-NEXT:      "MangledName": "",
// CHECK-NEXT:      "Name": "Fizz",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "TagType": "struct",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "Access": "private",
// CHECK-NEXT:      "End": true,
// CHECK-NEXT:      "InfoType": "record",
// CHECK-NEXT:      "IsParent": false,
// CHECK-NEXT:      "IsTypedef": false,
// CHECK-NEXT:      "IsVirtual": true,
// CHECK-NEXT:      "MangledName": "",
// CHECK-NEXT:      "Name": "Virtual",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "TagType": "struct",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    }
// CHECK-NEXT:  ],
// CHECK:       "Parents": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "Name": "Bar",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "QualName": "Bar",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "End": true,
// CHECK-NEXT:      "Name": "Buzz",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "QualName": "Buzz",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    }
// CHECK-NEXT:  ],
// CHECK:       "VirtualParents": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "End": true,
// CHECK-NEXT:      "Name": "Virtual",
// CHECK-NEXT:      "Path": "GlobalNamespace",
// CHECK-NEXT:      "QualName": "Virtual",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    }
// CHECK-NEXT:  ]
