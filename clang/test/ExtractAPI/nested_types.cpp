// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api --pretty-sgf -triple arm64-apple-macosx \
// RUN:   -x c++-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
struct MyStruct {
    struct {
        int count;
    } counts[1];
};

/// expected-no-diagnostics

//--- reference.output.json.in
{
  "metadata": {
    "formatVersion": {
      "major": 0,
      "minor": 5,
      "patch": 3
    },
    "generator": "?"
  },
  "module": {
    "name": "",
    "platform": {
      "architecture": "arm64",
      "operatingSystem": {
        "minimumVersion": {
          "major": 11,
          "minor": 0,
          "patch": 0
        },
        "name": "macosx"
      },
      "vendor": "apple"
    }
  },
  "relationships": [
    {
      "kind": "memberOf",
      "source": "c:@S@MyStruct@S@input.h@22",
      "target": "c:@S@MyStruct",
      "targetFallback": "MyStruct"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@MyStruct@S@input.h@22@FI@count",
      "target": "c:@S@MyStruct@S@input.h@22",
      "targetFallback": ""
    },
    {
      "kind": "memberOf",
      "source": "c:@S@MyStruct@FI@counts",
      "target": "c:@S@MyStruct",
      "targetFallback": "MyStruct"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "struct"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyStruct"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@MyStruct"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "c++.struct"
      },
      "location": {
        "position": {
          "character": 7,
          "line": 0
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyStruct"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyStruct"
          }
        ],
        "title": "MyStruct"
      },
      "pathComponents": [
        "MyStruct"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "struct"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@MyStruct@S@input.h@22"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "c++.struct"
      },
      "location": {
        "position": {
          "character": 4,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": ""
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": ""
          }
        ],
        "title": ""
      },
      "pathComponents": [
        "MyStruct",
        ""
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "count"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@MyStruct@S@input.h@22@FI@count"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c++.property"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "count"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "count"
          }
        ],
        "title": "count"
      },
      "pathComponents": [
        "MyStruct",
        "",
        "count"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "struct"
        },
        {
          "kind": "text",
          "spelling": " { ... } "
        },
        {
          "kind": "identifier",
          "spelling": "counts"
        },
        {
          "kind": "text",
          "spelling": "["
        },
        {
          "kind": "number",
          "spelling": "1"
        },
        {
          "kind": "text",
          "spelling": "];"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@MyStruct@FI@counts"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c++.property"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "counts"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "counts"
          }
        ],
        "title": "counts"
      },
      "pathComponents": [
        "MyStruct",
        "counts"
      ]
    }
  ]
}
