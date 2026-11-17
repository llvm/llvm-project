// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/class-specialization.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV7MyClass.json --check-prefix=BASE
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV7MyClassIiE.json --check-prefix=SPECIALIZATION
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json --check-prefix=JSON-NAMESPACE

// BASE:       "MangledName": "_ZTV7MyClass",
// BASE-NEXT:  "Name": "MyClass",
// BASE-NEXT:  "Namespace": [
// BASE-NEXT:    "GlobalNamespace"
// BASE-NEXT:  ],
// BASE-NEXT:  "Path": "GlobalNamespace",
// BASE-NEXT:  "TagType": "struct",
// BASE-NEXT:  "Template": {
// BASE-NEXT:    "Parameters": [
// BASE-NEXT:      {
// BASE-NEXT:        "End": true,
// BASE-NEXT:        "Param": "typename T"
// BASE-NEXT:      }
// BASE-NEXT:    ],
// BASE-NEXT:    "VerticalDisplay": false
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
// SPECIALIZATION-NEXT:        {
// SPECIALIZATION-NEXT:          "Param": "int",
// SPECIALIZATION-NEXT:          "SpecParamEnd": true
// SPECIALIZATION-NEXT:        }
// SPECIALIZATION-NEXT:      ],
// SPECIALIZATION-NEXT:      "SpecializationOf": "{{[0-9A-F]*}}",
// SPECIALIZATION-NEXT:      "VerticalDisplay": false
// SPECIALIZATION-NEXT:    }
// SPECIALIZATION-NEXT:  },

// JSON-NAMESPACE:       "Records": [
// JSON-NAMESPACE-NEXT:    {
// JSON-NAMESPACE-NEXT:      "DocumentationFileName": "_ZTV7MyClass",
// JSON-NAMESPACE-NEXT:      "Name": "MyClass",
// JSON-NAMESPACE-NEXT:      "Path": "GlobalNamespace",
// JSON-NAMESPACE-NEXT:      "QualName": "MyClass",
// JSON-NAMESPACE-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// JSON-NAMESPACE-NEXT:    },
// JSON-NAMESPACE-NEXT:    {
// JSON-NAMESPACE-NEXT:      "DocumentationFileName": "_ZTV7MyClassIiE",
// JSON-NAMESPACE-NEXT:      "End": true,
// JSON-NAMESPACE-NEXT:      "Name": "MyClass",
// JSON-NAMESPACE-NEXT:      "Path": "GlobalNamespace",
// JSON-NAMESPACE-NEXT:      "QualName": "MyClass",
// JSON-NAMESPACE-NEXT:      "Specialization": {
// JSON-NAMESPACE-NEXT:        "Parameters": [
// JSON-NAMESPACE-NEXT:          {
// JSON-NAMESPACE-NEXT:            "Param": "int",
// JSON-NAMESPACE-NEXT:            "SpecParamEnd": true
// JSON-NAMESPACE-NEXT:          }
// JSON-NAMESPACE-NEXT:        ],
// JSON-NAMESPACE-NEXT:        "SpecializationOf": "{{([0-9A-F]{40})}}",
// JSON-NAMESPACE-NEXT:        "VerticalDisplay": false
// JSON-NAMESPACE-NEXT:      },
// JSON-NAMESPACE-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// JSON-NAMESPACE-NEXT:    }
// JSON-NAMESPACE-NEXT:  ]
