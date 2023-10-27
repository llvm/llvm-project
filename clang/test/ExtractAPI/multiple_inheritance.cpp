
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
class Fizz {};

class Foo : public Fizz {};

class Bar : public Fizz {};

class FooBar : public Foo, public Bar{};
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
      "kind": "inheritsFrom",
      "source": "c:@S@Foo",
      "target": "c:@S@Fizz",
      "targetFallback": "Fizz"
    },
    {
      "kind": "inheritsFrom",
      "source": "c:@S@Bar",
      "target": "c:@S@Fizz",
      "targetFallback": "Fizz"
    },
    {
      "kind": "inheritsFrom",
      "source": "c:@S@FooBar",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "inheritsFrom",
      "source": "c:@S@FooBar",
      "target": "c:@S@Bar",
      "targetFallback": "Bar"
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
          "spelling": "Fizz"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Fizz"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "c++.class"
      },
      "location": {
        "position": {
          "character": 7,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Fizz"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Fizz"
          }
        ],
        "title": "Fizz"
      },
      "pathComponents": [
        "Fizz"
      ]
    },
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
          "character": 7,
          "line": 3
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
          "kind": "keyword",
          "spelling": "class"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Bar"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Bar"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "c++.class"
      },
      "location": {
        "position": {
          "character": 7,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Bar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Bar"
          }
        ],
        "title": "Bar"
      },
      "pathComponents": [
        "Bar"
      ]
    },
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
          "spelling": "FooBar"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@FooBar"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "c++.class"
      },
      "location": {
        "position": {
          "character": 7,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "FooBar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "FooBar"
          }
        ],
        "title": "FooBar"
      },
      "pathComponents": [
        "FooBar"
      ]
    }
  ]
}
