// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 %t/main.c --emit-symbol-graph=%t/SymbolGraphs --product-name=basicfile -triple=x86_64-apple-macosx12.0.0

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/SymbolGraphs/main.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- main.c
#define TESTMACRO1 2
#define TESTMARCRO2 5

int main ()
{
  return 0;
}


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
    "name": "basicfile",
    "platform": {
      "architecture": "x86_64",
      "operatingSystem": {
        "name": "macosx"
      },
      "vendor": "apple"
    }
  },
  "relationships": [],
  "symbols": [
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
          "spelling": "main"
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
        "interfaceLanguage": "c",
        "precise": "c:@F@main"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 4
        },
        "uri": "file://INPUT_DIR/main.c"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "main"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "main"
          }
        ],
        "title": "main"
      },
      "pathComponents": [
        "main"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "#define"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "TESTMACRO1"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:main.c@8@macro@TESTMACRO1"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "c.macro"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 1
        },
        "uri": "file://INPUT_DIR/main.c"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "TESTMACRO1"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "TESTMACRO1"
          }
        ],
        "title": "TESTMACRO1"
      },
      "pathComponents": [
        "TESTMACRO1"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "#define"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "TESTMARCRO2"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:main.c@29@macro@TESTMARCRO2"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "c.macro"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 2
        },
        "uri": "file://INPUT_DIR/main.c"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "TESTMARCRO2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "TESTMARCRO2"
          }
        ],
        "title": "TESTMARCRO2"
      },
      "pathComponents": [
        "TESTMARCRO2"
      ]
    }
  ]
}
