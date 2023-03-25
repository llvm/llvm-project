// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
             // RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api -triple arm64-apple-macosx -x objective-c-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
             // RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json


//--- input.h
@interface Foo
- (instancetype) init;
- (id) reset;
@end
// expected-no-diagnostics


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
      "source": "c:objc(cs)Foo(im)init",
      "target": "c:objc(cs)Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Foo(im)reset",
      "target": "c:objc(cs)Foo",
      "targetFallback": "Foo"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@interface"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Foo"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Foo"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "objective-c.class"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 1
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
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "keyword",
          "spelling": "instancetype"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "init"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "keyword",
            "spelling": "instancetype"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Foo(im)init"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "init"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "init"
          }
        ],
        "title": "init"
      },
      "pathComponents": [
        "Foo",
        "init"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "keyword",
          "spelling": "id"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "reset"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "keyword",
            "spelling": "id"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Foo(im)reset"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "reset"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "reset"
          }
        ],
        "title": "reset"
      },
      "pathComponents": [
        "Foo",
        "reset"
      ]
    }
  ]
}
