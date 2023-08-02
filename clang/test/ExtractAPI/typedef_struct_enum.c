// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang -extract-api -target arm64-apple-macosx \
// RUN: %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
typedef struct Test {
} Test;

typedef enum Test2 {
  simple
} Test2;

struct Foo;
typedef struct Foo TypedefedFoo;
struct Foo {
    int bar;
};

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
      "source": "c:@E@Test2@simple",
      "target": "c:@E@Test2",
      "targetFallback": "Test2"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@FI@bar",
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
          "spelling": "typedef"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "keyword",
          "spelling": "enum"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Test2"
        },
        {
          "kind": "text",
          "spelling": ": "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": " { ... } "
        },
        {
          "kind": "identifier",
          "spelling": "Test2"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Test2"
      },
      "kind": {
        "displayName": "Enumeration",
        "identifier": "c.enum"
      },
      "location": {
        "position": {
          "character": 14,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Test2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Test2"
          }
        ],
        "title": "Test2"
      },
      "pathComponents": [
        "Test2"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "simple"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Test2@simple"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 3,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "simple"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "simple"
          }
        ],
        "title": "simple"
      },
      "pathComponents": [
        "Test2",
        "simple"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "typedef"
        },
        {
          "kind": "text",
          "spelling": " "
        },
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
          "spelling": "Test"
        },
        {
          "kind": "text",
          "spelling": " { ... } "
        },
        {
          "kind": "identifier",
          "spelling": "Test"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Test"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "c.struct"
      },
      "location": {
        "position": {
          "character": 16,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Test"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Test"
          }
        ],
        "title": "Test"
      },
      "pathComponents": [
        "Test"
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
        "interfaceLanguage": "c",
        "precise": "c:@S@Foo"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "c.struct"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 10
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
          "spelling": "bar"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Foo@FI@bar"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 11
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "bar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "bar"
          }
        ],
        "title": "bar"
      },
      "pathComponents": [
        "Foo",
        "bar"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "typedef"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "keyword",
          "spelling": "struct"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:@S@Foo",
          "spelling": "Foo"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "TypedefedFoo"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:input.h@T@TypedefedFoo"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "c.typealias"
      },
      "location": {
        "position": {
          "character": 20,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "TypedefedFoo"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "TypedefedFoo"
          }
        ],
        "title": "TypedefedFoo"
      },
      "pathComponents": [
        "TypedefedFoo"
      ],
      "type": "c:@S@Foo"
    }
  ]
}
