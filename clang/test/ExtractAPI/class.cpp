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
class Foo {
private:
  int a;
  mutable int b;

protected:
  int c;

public:
  int d;
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
      "source": "c:@S@Foo@FI@a",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@FI@b",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@FI@c",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@FI@d",
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
          "spelling": "a"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo@FI@a"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c++.property"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "a"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "a"
          }
        ],
        "title": "a"
      },
      "pathComponents": [
        "Foo",
        "a"
      ]
    },
    {
      "accessLevel": "private",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "mutable"
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
          "spelling": "b"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo@FI@b"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c++.property"
      },
      "location": {
        "position": {
          "character": 14,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "b"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "b"
          }
        ],
        "title": "b"
      },
      "pathComponents": [
        "Foo",
        "b"
      ]
    },
    {
      "accessLevel": "protected",
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
          "spelling": "c"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo@FI@c"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c++.property"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "c"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "c"
          }
        ],
        "title": "c"
      },
      "pathComponents": [
        "Foo",
        "c"
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
          "spelling": "d"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Foo@FI@d"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c++.property"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "d"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "d"
          }
        ],
        "title": "d"
      },
      "pathComponents": [
        "Foo",
        "d"
      ]
    }
  ]
}
