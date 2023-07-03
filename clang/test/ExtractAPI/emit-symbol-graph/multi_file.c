// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.main.json.in >> %t/reference.main.json
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.test.json.in >> %t/reference.test.json
// RUN: %clang_cc1 %t/test.c %t/main.c --emit-symbol-graph=%t/SymbolGraphs --product-name=multifile_test -triple=x86_64-apple-macosx12.0.0

// Test main.json
// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/SymbolGraphs/main.json > %t/output-normalized.json
// RUN: diff %t/reference.main.json %t/output-normalized.json

// Test test.json
// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/SymbolGraphs/test.json > %t/output-normalized.json
// RUN: diff %t/reference.test.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- test.h
#ifndef TEST_H
#define TEST_H

#define testmarcro1 32
#define testmacro2 42

int testfunc (int param1, int param2);
void testfunc2 ();
#endif /* TEST_H */

//--- test.c
#include "test.h"

int testfunc(int param1, int param2) { return param1 + param2; }

void testfunc2() {}

//--- main.c
#include "test.h"

int main ()
{
  testfunc2();
  return 0;
}

//--- reference.main.json.in
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
    "name": "multifile_test",
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
          "spelling": "testfunc"
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
          "spelling": "param1"
        },
        {
          "kind": "text",
          "spelling": ", "
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
          "spelling": "param2"
        },
        {
          "kind": "text",
          "spelling": ");"
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
                "spelling": "param1"
              }
            ],
            "name": "param1"
          },
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
                "spelling": "param2"
              }
            ],
            "name": "param2"
          }
        ],
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
        "precise": "c:@F@testfunc"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 7
        },
        "uri": "file://INPUT_DIR/test.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "testfunc"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "testfunc"
          }
        ],
        "title": "testfunc"
      },
      "pathComponents": [
        "testfunc"
      ]
    },
    {
      "accessLevel": "public",
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
          "spelling": "testfunc2"
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
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@F@testfunc2"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 8
        },
        "uri": "file://INPUT_DIR/test.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "testfunc2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "testfunc2"
          }
        ],
        "title": "testfunc2"
      },
      "pathComponents": [
        "testfunc2"
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
          "line": 3
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
          "spelling": "testmarcro1"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:test.h@39@macro@testmarcro1"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "c.macro"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 4
        },
        "uri": "file://INPUT_DIR/test.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "testmarcro1"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "testmarcro1"
          }
        ],
        "title": "testmarcro1"
      },
      "pathComponents": [
        "testmarcro1"
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
          "spelling": "testmacro2"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:test.h@62@macro@testmacro2"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "c.macro"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 5
        },
        "uri": "file://INPUT_DIR/test.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "testmacro2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "testmacro2"
          }
        ],
        "title": "testmacro2"
      },
      "pathComponents": [
        "testmacro2"
      ]
    }
  ]
}
//--- reference.test.json.in
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
    "name": "multifile_test",
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
          "spelling": "testfunc"
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
          "spelling": "param1"
        },
        {
          "kind": "text",
          "spelling": ", "
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
          "spelling": "param2"
        },
        {
          "kind": "text",
          "spelling": ");"
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
                "spelling": "param1"
              }
            ],
            "name": "param1"
          },
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
                "spelling": "param2"
              }
            ],
            "name": "param2"
          }
        ],
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
        "precise": "c:@F@testfunc"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 7
        },
        "uri": "file://INPUT_DIR/test.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "testfunc"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "testfunc"
          }
        ],
        "title": "testfunc"
      },
      "pathComponents": [
        "testfunc"
      ]
    },
    {
      "accessLevel": "public",
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
          "spelling": "testfunc2"
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
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@F@testfunc2"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 8
        },
        "uri": "file://INPUT_DIR/test.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "testfunc2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "testfunc2"
          }
        ],
        "title": "testfunc2"
      },
      "pathComponents": [
        "testfunc2"
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
          "spelling": "testmarcro1"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:test.h@39@macro@testmarcro1"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "c.macro"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 4
        },
        "uri": "file://INPUT_DIR/test.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "testmarcro1"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "testmarcro1"
          }
        ],
        "title": "testmarcro1"
      },
      "pathComponents": [
        "testmarcro1"
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
          "spelling": "testmacro2"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:test.h@62@macro@testmacro2"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "c.macro"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 5
        },
        "uri": "file://INPUT_DIR/test.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "testmacro2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "testmacro2"
          }
        ],
        "title": "testmacro2"
      },
      "pathComponents": [
        "testmacro2"
      ]
    }
  ]
}
