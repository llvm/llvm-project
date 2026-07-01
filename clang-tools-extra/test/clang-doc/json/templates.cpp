// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --doxygen --executor=standalone %S/../Inputs/templates.cpp -output=%t/docs --format=html
// RUN: FileCheck %s < %t/docs/json/GlobalNamespace/index.json --check-prefix=JSON

// JSON:           "Name": "ParamPackFunction",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "args",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "T...",
// JSON-NEXT:            "QualName": "T...",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": true,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "void",
// JSON-NEXT:        "QualName": "void",
// JSON-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:      },
// JSON-NEXT:      "Template": {
// JSON-NEXT:        "Parameters": [
// JSON-NEXT:          {
// JSON-NEXT:            "End": true,
// JSON-NEXT:            "Param": "class... T"
// JSON-NEXT:          }
// JSON-NEXT:        ],
// JSON-NEXT:        "VerticalDisplay": false
// JSON-NEXT:      },

// JSON:           "Name": "function",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "x",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "T",
// JSON-NEXT:            "QualName": "T",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": true,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "void",
// JSON-NEXT:        "QualName": "void",
// JSON-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:      },
// JSON-NEXT:      "Template": {
// JSON-NEXT:        "Parameters": [
// JSON-NEXT:          {
// JSON-NEXT:            "Param": "typename T"
// JSON-NEXT:          },
// JSON-NEXT:          {
// JSON-NEXT:            "End": true,
// JSON-NEXT:            "Param": "int U = 1"
// JSON-NEXT:          }
// JSON-NEXT:        ],
// JSON-NEXT:        "VerticalDisplay": false
// JSON-NEXT:      }

// JSON:           "Name": "longFunction",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "a",
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "A",
// JSON-NEXT:            "QualName": "A",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        },
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "b",
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "B",
// JSON-NEXT:            "QualName": "B",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        },
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "c",
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "C",
// JSON-NEXT:            "QualName": "C",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        },
// JSON-NEXT:       {
// JSON-NEXT:          "Name": "d",
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "D",
// JSON-NEXT:            "QualName": "D",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        },
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "e",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "E",
// JSON-NEXT:            "QualName": "E",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": true,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "void",
// JSON-NEXT:        "QualName": "void",
// JSON-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:      },
// JSON-NEXT:      "Template": {
// JSON-NEXT:          "Parameters": [
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "typename A"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "typename B"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "typename C"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "typename D"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "End": true,
// JSON-NEXT:              "Param": "typename E"
// JSON-NEXT:            }
// JSON-NEXT:          ],
// JSON-NEXT:          "VerticalDisplay": true
// JSON-NEXT:        },
// JSON-NEXT:        "USR": "{{([0-9A-F]{40})}}",
// JSON-NEXT:        "VerticalDisplay": true
// JSON-NEXT:      }

// JSON:           "Name": "function",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "x",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "bool",
// JSON-NEXT:            "QualName": "bool",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": true,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "void",
// JSON-NEXT:        "QualName": "void",
// JSON-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:      },
// JSON-NEXT:      "Template": {
// JSON-NEXT:        "Specialization": {
// JSON-NEXT:          "Parameters": [
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "bool"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "0",
// JSON-NEXT:              "SpecParamEnd": true
// JSON-NEXT:            }
// JSON-NEXT:          ],
// JSON-NEXT:          "SpecializationOf": "{{([0-9A-F]{40})}}",
// JSON-NEXT:          "VerticalDisplay": false
// JSON-NEXT:        }
// JSON-NEXT:      }

// JSON:           "Name": "func_with_tuple_param",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "t",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "tuple",
// JSON-NEXT:            "Path": "GlobalNamespace",
// JSON-NEXT:            "QualName": "tuple<int, int, bool>",
// JSON-NEXT:            "USR": "{{([0-9A-F]{40})}}"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": false,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "tuple",
// JSON-NEXT:        "QualName": "tuple<int, int, bool>",
// JSON-NEXT:        "USR": "{{([0-9A-F]{40})}}"
// JSON-NEXT:      }
