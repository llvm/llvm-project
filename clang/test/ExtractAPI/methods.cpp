// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api -triple arm64-apple-macosx \
// RUN:   -x c++-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
class Foo {
  int getCount();

  void setLength(int length) noexcept;

public:
  static double getFoo();

protected:
  constexpr int getBar() const;
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
      "source": "c:@S@Foo@F@getCount#",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@F@setLength#I#",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@F@getBar#1",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@F@getFoo#S",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "class"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Foo"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "c++.class"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 0
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Foo"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Foo"
          }
        ],
        "title": "Foo"
      },
      "pathComponents": [
        "Foo"
      ]
    },
    {
      "accessLevel": "private",
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
          "spelling": "getCount"
        },
        {
          "kind": "text",
          "spelling": "();"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:I",
            "spelling": "int"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo@F@getCount#"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "c++.method"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "getCount"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "getCount"
          }
        ],
        "title": "getCount"
      },
      "pathComponents": [
        "Foo",
        "getCount"
      ]
    },
    {
      "accessLevel": "private",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "setLength"
        },
        {
          "kind": "text",
          "spelling": "("
        },
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
          "kind": "internalParam",
          "spelling": "length"
        },
        {
          "kind": "text",
          "spelling": ")"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "keyword",
          "spelling": "noexcept"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
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
                "kind": "internalParam",
                "spelling": "length"
              }
            ],
            "name": "length"
          }
        ],
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo@F@setLength#I#"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "c++.method"
      },
      "location": {
        "position": {
          "character": 7,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "setLength"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "setLength"
          }
        ],
        "title": "setLength"
      },
      "pathComponents": [
        "Foo",
        "setLength"
      ]
    },
    {
      "accessLevel": "protected",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "constexpr"
        },
        {
          "kind": "text",
          "spelling": " "
        },
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
          "spelling": "getBar"
        },
        {
          "kind": "text",
          "spelling": "() "
        },
        {
          "kind": "keyword",
          "spelling": "const"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:I",
            "spelling": "int"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo@F@getBar#1"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "c++.method"
      },
      "location": {
        "position": {
          "character": 16,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "getBar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "getBar"
          }
        ],
        "title": "getBar"
      },
      "pathComponents": [
        "Foo",
        "getBar"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "static"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:d",
          "spelling": "double"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "getFoo"
        },
        {
          "kind": "text",
          "spelling": "();"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:d",
            "spelling": "double"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo@F@getFoo#S"
      },
      "kind": {
        "displayName": "Static Method",
        "identifier": "c++.type.method"
      },
      "location": {
        "position": {
          "character": 16,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "getFoo"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "getFoo"
          }
        ],
        "title": "getFoo"
      },
      "pathComponents": [
        "Foo",
        "getFoo"
      ]
    }
  ]
}
