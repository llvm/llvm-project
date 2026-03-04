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
class Outer {
  class Inner1 {
    class DeeplyNested {
      int value;
    };
  };

  struct Inner2 {
    enum Status { READY, PENDING };
  };
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
      "source": "c:@S@Outer@S@Inner1",
      "target": "c:@S@Outer",
      "targetFallback": "Outer"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Outer@S@Inner1@S@DeeplyNested",
      "target": "c:@S@Outer@S@Inner1",
      "targetFallback": "Inner1"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Outer@S@Inner1@S@DeeplyNested@FI@value",
      "target": "c:@S@Outer@S@Inner1@S@DeeplyNested",
      "targetFallback": "DeeplyNested"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Outer@S@Inner2",
      "target": "c:@S@Outer",
      "targetFallback": "Outer"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Outer@S@Inner2@E@Status",
      "target": "c:@S@Outer@S@Inner2",
      "targetFallback": "Inner2"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Outer@S@Inner2@E@Status@READY",
      "target": "c:@S@Outer@S@Inner2@E@Status",
      "targetFallback": "Status"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Outer@S@Inner2@E@Status@PENDING",
      "target": "c:@S@Outer@S@Inner2@E@Status",
      "targetFallback": "Status"
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
          "spelling": "Outer"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Outer"
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
            "spelling": "Outer"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Outer"
          }
        ],
        "title": "Outer"
      },
      "pathComponents": [
        "Outer"
      ]
    },
    {
      "accessLevel": "private",
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
          "spelling": "Inner1"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Outer@S@Inner1"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "c++.class"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Inner1"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Inner1"
          }
        ],
        "title": "Inner1"
      },
      "pathComponents": [
        "Outer",
        "Inner1"
      ]
    },
    {
      "accessLevel": "private",
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
          "spelling": "DeeplyNested"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Outer@S@Inner1@S@DeeplyNested"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "c++.class"
      },
      "location": {
        "position": {
          "character": 10,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "DeeplyNested"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "DeeplyNested"
          }
        ],
        "title": "DeeplyNested"
      },
      "pathComponents": [
        "Outer",
        "Inner1",
        "DeeplyNested"
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
          "spelling": "value"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Outer@S@Inner1@S@DeeplyNested@FI@value"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c++.property"
      },
      "location": {
        "position": {
          "character": 10,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "value"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "value"
          }
        ],
        "title": "value"
      },
      "pathComponents": [
        "Outer",
        "Inner1",
        "DeeplyNested",
        "value"
      ]
    },
    {
      "accessLevel": "private",
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
          "spelling": "Inner2"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Outer@S@Inner2"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "c++.struct"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Inner2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Inner2"
          }
        ],
        "title": "Inner2"
      },
      "pathComponents": [
        "Outer",
        "Inner2"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
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
          "spelling": "Status"
        },
        {
          "kind": "text",
          "spelling": " : "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Outer@S@Inner2@E@Status"
      },
      "kind": {
        "displayName": "Enumeration",
        "identifier": "c++.enum"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 8
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Status"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Status"
          }
        ],
        "title": "Status"
      },
      "pathComponents": [
        "Outer",
        "Inner2",
        "Status"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "READY"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Outer@S@Inner2@E@Status@READY"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c++.enum.case"
      },
      "location": {
        "position": {
          "character": 18,
          "line": 8
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "READY"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "READY"
          }
        ],
        "title": "READY"
      },
      "pathComponents": [
        "Outer",
        "Inner2",
        "Status",
        "READY"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "PENDING"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@S@Outer@S@Inner2@E@Status@PENDING"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c++.enum.case"
      },
      "location": {
        "position": {
          "character": 25,
          "line": 8
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "PENDING"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "PENDING"
          }
        ],
        "title": "PENDING"
      },
      "pathComponents": [
        "Outer",
        "Inner2",
        "Status",
        "PENDING"
      ]
    }
  ]
}
