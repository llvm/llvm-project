// RUN: %clang_cc1 -std=c++2a -triple x86_64-linux-gnu -fcxx-exceptions -ast-dump=json %s | FileCheck %s

namespace n {
void function() {}
int Variable;
}
using n::function;
using n::Variable;
void TestFunction() {
  void (*f)() = &function;
  Variable = 4;
}

void TestCatch1() {
  try {
  }
  catch (int x) {
  }
}

void TestCatch2() {
  try {
  }
  catch (...) {
  }
}

void TestAllocationExprs() {
  int *p;
  p = new int;
  delete p;
  p = new int[2];
  delete[] p;
  p = ::new int;
  ::delete p;
}

// Don't crash on dependent exprs that haven't been resolved yet.
template <typename T>
void TestDependentAllocationExpr() {
  T *p = new T;
  delete p;
}

template <typename T>
class DependentScopeMemberExprWrapper {
  T member;
};

template <typename T>
void TestDependentScopeMemberExpr() {
  DependentScopeMemberExprWrapper<T> obj;
  obj.member = T();
  (&obj)->member = T();
}

union U {
  int i;
  long l;
};

void TestUnionInitList()
{
  U us[3] = {1};
}

void TestSwitch(int i) {
  switch (int a; i)
    ;
}

void TestIf(bool b) {
  if (const int i = 12; i)
    ;

  if constexpr (sizeof(b) == 1)
    ;

  if constexpr (sizeof(b) == 1)
    ;
  else
    ;
}

struct Container {
  int *begin() const;
  int *end() const;
};

void TestIteration() {
  for (int i = 0; int j = i; ++i)
    ;

  int vals[10];
  for (int v : vals)
    ;

  Container C;
  for (int v : C)
    ;

  for (int a; int v : vals)
    ;
}


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 4
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 4
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 18,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 4
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "isUsed": true,
// CHECK-NEXT:  "name": "function",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 17,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 4
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 18,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 4
// CHECK-NEXT:     }
// CHECK-NEXT:    }
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "UsingDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 10,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 7
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 7
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 10,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 7
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "n::function"
// CHECK-NEXT: }


// CHECK:  "kind": "UsingShadowDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 10,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 7
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 10,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 7
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 10,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 7
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "target": {
// CHECK-NEXT:   "id": "0x{{.*}}",
// CHECK-NEXT:   "kind": "FunctionDecl",
// CHECK-NEXT:   "name": "function",
// CHECK-NEXT:   "type": {
// CHECK-NEXT:    "qualType": "void ()"
// CHECK-NEXT:   }
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "UsingDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 10,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 8
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 8
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 10,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 8
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "n::Variable"
// CHECK-NEXT: }


// CHECK:  "kind": "UsingShadowDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 10,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 8
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 10,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 8
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 10,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 8
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "target": {
// CHECK-NEXT:   "id": "0x{{.*}}",
// CHECK-NEXT:   "kind": "VarDecl",
// CHECK-NEXT:   "name": "Variable",
// CHECK-NEXT:   "type": {
// CHECK-NEXT:    "qualType": "int"
// CHECK-NEXT:   }
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 9
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 9
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 12
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestFunction",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 21,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 9
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 12
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 10
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 26,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 10
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 10,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 10
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 10
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 10
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "name": "f",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (*)()"
// CHECK-NEXT:        },
// CHECK-NEXT:        "init": "c",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "UnaryOperator",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 10
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 10
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void (*)()"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "isPostfix": false,
// CHECK-NEXT:          "opcode": "&",
// CHECK-NEXT:          "canOverflow": false,
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 10
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 10
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "void ()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "FunctionDecl",
// CHECK-NEXT:             "name": "function",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "void ()"
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "foundReferencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "UsingShadowDecl",
// CHECK-NEXT:             "name": "function"
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "BinaryOperator",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 11
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 14,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 11
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "opcode": "=",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclRefExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 11
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 11
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "referencedDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "VarDecl",
// CHECK-NEXT:         "name": "Variable",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "int"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "foundReferencedDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "UsingShadowDecl",
// CHECK-NEXT:         "name": "Variable"
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "IntegerLiteral",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 11
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 11
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "value": "4"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 14
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 14
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 19
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestCatch1",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 19,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 14
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXTryStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 15
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 18
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CompoundStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 15
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 16
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXCatchStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 17
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 18
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 14,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 17
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 17
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 17
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "name": "x",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          }
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 17
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 18
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 21
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 21
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 26
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestCatch2",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 19,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 21
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 26
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXTryStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 22
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CompoundStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 22
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 23
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXCatchStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 24
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 25
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 15,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 24
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 25
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 28
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 28
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 36
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestAllocationExprs",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 28,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 28
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 36
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 29
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 29
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 8,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 29
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 29
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 8,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 29
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "isUsed": true,
// CHECK-NEXT:        "name": "p",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "BinaryOperator",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 30
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 30
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int *"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "opcode": "=",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclRefExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 30
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 30
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "referencedDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "VarDecl",
// CHECK-NEXT:         "name": "p",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "int *"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 30
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 30
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(unsigned long)"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXDeleteExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 31
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 10,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 31
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "operatorDeleteDecl": {
// CHECK-NEXT:       "id": "0x{{.*}}",
// CHECK-NEXT:       "kind": "FunctionDecl",
// CHECK-NEXT:       "name": "operator delete",
// CHECK-NEXT:       "type": {
// CHECK-NEXT:        "qualType": "void (void *) noexcept"
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 31
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 31
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 31
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 31
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "p",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "BinaryOperator",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 32
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 16,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 32
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int *"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "opcode": "=",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclRefExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 32
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 32
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "referencedDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "VarDecl",
// CHECK-NEXT:         "name": "p",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "int *"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 32
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 32
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isArray": true,
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new[]",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(unsigned long)"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 15,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 32
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 15,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 32
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "unsigned long"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "IntegralCast",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 15,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 32
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 15,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 32
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "2"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXDeleteExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 33
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 12,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 33
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "isArray": true,
// CHECK-NEXT:      "isArrayAsWritten": true,
// CHECK-NEXT:      "operatorDeleteDecl": {
// CHECK-NEXT:       "id": "0x{{.*}}",
// CHECK-NEXT:       "kind": "FunctionDecl",
// CHECK-NEXT:       "name": "operator delete[]",
// CHECK-NEXT:       "type": {
// CHECK-NEXT:        "qualType": "void (void *) noexcept"
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 12,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 33
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 12,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 33
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 12,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 33
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 12,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 33
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "p",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "BinaryOperator",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 13,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int *"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "opcode": "=",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclRefExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 34
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 34
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "referencedDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "VarDecl",
// CHECK-NEXT:         "name": "p",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "int *"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 34
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 13,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 34
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isGlobal": true,
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(unsigned long)"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXDeleteExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 35
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 12,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 35
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "isGlobal": true,
// CHECK-NEXT:      "operatorDeleteDecl": {
// CHECK-NEXT:       "id": "0x{{.*}}",
// CHECK-NEXT:       "kind": "FunctionDecl",
// CHECK-NEXT:       "name": "operator delete",
// CHECK-NEXT:       "type": {
// CHECK-NEXT:        "qualType": "void (void *) noexcept"
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 12,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 35
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 12,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 35
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 12,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 35
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 12,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 35
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "p",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "isUsed": true,
// CHECK-NEXT:  "name": "operator new",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void *(unsigned long)"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "unsigned long"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VisibilityAttr",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "implicit": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "name": "operator new",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void *(unsigned long, std::align_val_t)"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "unsigned long"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "std::align_val_t"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VisibilityAttr",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "implicit": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "isUsed": true,
// CHECK-NEXT:  "name": "operator new[]",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void *(unsigned long)"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "unsigned long"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VisibilityAttr",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "implicit": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "name": "operator new[]",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void *(unsigned long, std::align_val_t)"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "unsigned long"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "std::align_val_t"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VisibilityAttr",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "implicit": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "isUsed": true,
// CHECK-NEXT:  "name": "operator delete",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void *) noexcept"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void *"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VisibilityAttr",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "implicit": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "name": "operator delete",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void *, std::align_val_t) noexcept"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void *"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "std::align_val_t"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VisibilityAttr",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "implicit": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "isUsed": true,
// CHECK-NEXT:  "name": "operator delete[]",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void *) noexcept"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void *"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VisibilityAttr",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "implicit": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  },
// CHECK-NEXT:  "isImplicit": true,
// CHECK-NEXT:  "name": "operator delete[]",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void *, std::align_val_t) noexcept"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void *"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {},
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "isImplicit": true,
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "std::align_val_t"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VisibilityAttr",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {},
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    },
// CHECK-NEXT:    "implicit": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionTemplateDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 40
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 39
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 43
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestDependentAllocationExpr",
// CHECK-NEXT:  "templateParams": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "TemplateTypeParmDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 20,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 39
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 11,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 39
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 20,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 39
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isReferenced": true,
// CHECK-NEXT:    "name": "T",
// CHECK-NEXT:    "tagUsed": "typename",
// CHECK-NEXT:    "depth": 0,
// CHECK-NEXT:    "index": 0
// CHECK-NEXT:   }
// CHECK-NEXT:  ],
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "FunctionDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 40
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 40
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 43
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "name": "TestDependentAllocationExpr",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CompoundStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 36,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 40
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 1,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 43
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 41
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 41
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 6,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 41
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 41
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 41
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isReferenced": true,
// CHECK-NEXT:          "name": "p",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "T *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXNewExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 10,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 41
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 41
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "T *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXDeleteExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 42
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 42
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 42
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 42
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "T *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "p",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "T *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionTemplateDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 51
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 50
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 55
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestDependentScopeMemberExpr",
// CHECK-NEXT:  "templateParams": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "TemplateTypeParmDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 20,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 50
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 11,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 50
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 20,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 50
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isReferenced": true,
// CHECK-NEXT:    "name": "T",
// CHECK-NEXT:    "tagUsed": "typename",
// CHECK-NEXT:    "depth": 0,
// CHECK-NEXT:    "index": 0
// CHECK-NEXT:   }
// CHECK-NEXT:  ],
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "FunctionDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 51
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 51
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 55
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "name": "TestDependentScopeMemberExpr",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CompoundStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 37,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 51
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 1,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 55
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 52
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 41,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 52
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 38,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 52
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 52
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 38,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 52
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isReferenced": true,
// CHECK-NEXT:          "name": "obj",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "DependentScopeMemberExprWrapper<T>"
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "BinaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 53
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 53
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<dependent type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "opcode": "=",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXDependentScopeMemberExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 53
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 53
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "<dependent type>"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 53
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 53
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "DependentScopeMemberExprWrapper<T>"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "obj",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "DependentScopeMemberExprWrapper<T>"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXUnresolvedConstructExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 53
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 53
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "T"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue"
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "BinaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 54
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 22,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 54
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<dependent type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "opcode": "=",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXDependentScopeMemberExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 54
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 11,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 54
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "<dependent type>"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ParenExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 54
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 54
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "<dependent type>"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "UnaryOperator",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 4,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 54
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 5,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 54
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "<dependent type>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "isPostfix": false,
// CHECK-NEXT:              "opcode": "&",
// CHECK-NEXT:              "canOverflow": false,
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "DeclRefExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 5,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 54
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 5,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 54
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "DependentScopeMemberExprWrapper<T>"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "lvalue",
// CHECK-NEXT:                "referencedDecl": {
// CHECK-NEXT:                 "id": "0x{{.*}}",
// CHECK-NEXT:                 "kind": "VarDecl",
// CHECK-NEXT:                 "name": "obj",
// CHECK-NEXT:                 "type": {
// CHECK-NEXT:                  "qualType": "DependentScopeMemberExprWrapper<T>"
// CHECK-NEXT:                 }
// CHECK-NEXT:                }
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXUnresolvedConstructExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 20,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 54
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 22,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 54
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "T"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue"
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 62
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 62
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 65
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestUnionInitList",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 63
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 64
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 16,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 64
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 5,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 64
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 64
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 64
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "name": "us",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "U [3]"
// CHECK-NEXT:        },
// CHECK-NEXT:        "init": "c",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "InitListExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 13,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 64
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 15,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 64
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "U [3]"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "array_filler": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "InitListExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 15,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 64
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 15,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 64
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "U"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue"
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "InitListExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 64
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 64
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "U"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "IntegerLiteral",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 64
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 64
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "value": "1"
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 67
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 67
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 70
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestSwitch",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (int)"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 21,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 67
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 17,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 67
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 21,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 67
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "i",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 24,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 67
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 70
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "SwitchStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 68
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 69
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "hasInit": true,
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 68
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 68
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 15,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 68
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 11,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 68
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 15,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 68
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "name": "a",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 68
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 68
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 68
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 68
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "i",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 69
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 69
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 72
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 72
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 83
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestIf",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (bool)"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 18,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 72
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 13,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 72
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 18,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 72
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isReferenced": true,
// CHECK-NEXT:    "name": "b",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "bool"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 21,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 72
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 83
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "IfStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 73
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 74
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "hasInit": true,
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 73
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 23,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 73
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 17,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 73
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 73
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 73
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isReferenced": true,
// CHECK-NEXT:          "name": "i",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "const int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 73
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 73
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "12"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 25,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 73
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 25,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 73
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "bool"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "IntegralToBoolean",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 25,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 73
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 25,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 73
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 25,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 73
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 25,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 73
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "const int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "i",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "const int"
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "nonOdrUseReason": "constant"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 74
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 74
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "IfStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 76
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 77
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "isConstexpr": true,
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ConstantExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 76
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 30,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 76
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "bool"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "BinaryOperator",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 76
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 30,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 76
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "bool"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "opcode": "==",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "UnaryExprOrTypeTraitExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 17,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 76
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 25,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 76
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "unsigned long"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "name": "sizeof",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "ParenExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 23,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 76
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 25,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 76
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "bool"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "DeclRefExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 24,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 76
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 24,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 76
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "bool"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "lvalue",
// CHECK-NEXT:                "referencedDecl": {
// CHECK-NEXT:                 "id": "0x{{.*}}",
// CHECK-NEXT:                 "kind": "ParmVarDecl",
// CHECK-NEXT:                 "name": "b",
// CHECK-NEXT:                 "type": {
// CHECK-NEXT:                  "qualType": "bool"
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "nonOdrUseReason": "unevaluated"
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 30,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 76
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 30,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 76
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "unsigned long"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "IntegralCast",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "IntegerLiteral",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 30,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 76
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 30,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 76
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "value": "1"
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 77
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 77
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "IfStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 79
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 82
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "hasElse": true,
// CHECK-NEXT:      "isConstexpr": true,
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ConstantExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 79
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 30,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 79
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "bool"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "BinaryOperator",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 79
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 30,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 79
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "bool"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "opcode": "==",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "UnaryExprOrTypeTraitExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 17,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 79
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 25,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 79
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "unsigned long"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "name": "sizeof",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "ParenExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 23,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 79
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 25,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 79
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "bool"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "DeclRefExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 24,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 79
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 24,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 79
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "bool"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "lvalue",
// CHECK-NEXT:                "referencedDecl": {
// CHECK-NEXT:                 "id": "0x{{.*}}",
// CHECK-NEXT:                 "kind": "ParmVarDecl",
// CHECK-NEXT:                 "name": "b",
// CHECK-NEXT:                 "type": {
// CHECK-NEXT:                  "qualType": "bool"
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "nonOdrUseReason": "unevaluated"
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 30,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 79
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 30,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 79
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "unsigned long"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "IntegralCast",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "IntegerLiteral",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 30,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 79
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 30,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 79
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "value": "1"
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 80
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 80
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 82
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 82
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 90
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 90
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 104
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestIteration",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 22,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 90
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 104
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ForStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 91
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 92
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 8,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 91
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 91
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 12,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 91
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 91
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 91
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "i",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 91
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 91
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "0"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 19,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 91
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 27,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 91
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 23,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 91
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 19,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 91
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 27,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 91
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "j",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 27,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 91
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 27,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 91
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 27,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 91
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 27,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 91
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "VarDecl",
// CHECK-NEXT:               "name": "i",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 23,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 91
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 23,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 91
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "bool"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "IntegralToBoolean",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 23,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 91
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 23,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 91
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 23,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 91
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 23,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 91
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "j",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "int"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "UnaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 30,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 91
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 32,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 91
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "isPostfix": false,
// CHECK-NEXT:        "opcode": "++",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 32,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 91
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 32,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 91
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "i",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 92
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 92
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 94
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 15,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 94
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 7,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 94
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 94
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 94
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "isUsed": true,
// CHECK-NEXT:        "name": "vals",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int [10]"
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXForRangeStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 95
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 96
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {},
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 16,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 95
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__range1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int (&)[10]"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int [10]"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "vals",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "int [10]"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 14,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 95
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__begin1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "ArrayToPointerDecay",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int [10]"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "VarDecl",
// CHECK-NEXT:               "name": "__range1",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int (&)[10]"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 14,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 95
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__end1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "BinaryOperator",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "opcode": "+",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "ImplicitCastExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int *"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "castKind": "ArrayToPointerDecay",
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "DeclRefExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 95
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 95
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "int [10]"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "lvalue",
// CHECK-NEXT:                "referencedDecl": {
// CHECK-NEXT:                 "id": "0x{{.*}}",
// CHECK-NEXT:                 "kind": "VarDecl",
// CHECK-NEXT:                 "name": "__range1",
// CHECK-NEXT:                 "type": {
// CHECK-NEXT:                  "qualType": "int (&)[10]"
// CHECK-NEXT:                 }
// CHECK-NEXT:                }
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "IntegerLiteral",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "long"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "value": "10"
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "BinaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "bool"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "opcode": "!=",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "int *",
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "__begin1",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "int *",
// CHECK-NEXT:              "qualType": "int *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "int *",
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "__end1",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "int *",
// CHECK-NEXT:              "qualType": "int *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "UnaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "desugaredQualType": "int *",
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "isPostfix": false,
// CHECK-NEXT:        "opcode": "++",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "__begin1",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "desugaredQualType": "int *",
// CHECK-NEXT:            "qualType": "int *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 8,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 20,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 95
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 12,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 95
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 95
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "name": "v",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "UnaryOperator",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "isPostfix": false,
// CHECK-NEXT:              "opcode": "*",
// CHECK-NEXT:              "canOverflow": false,
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ImplicitCastExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 95
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 95
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "desugaredQualType": "int *",
// CHECK-NEXT:                 "qualType": "int *"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "rvalue",
// CHECK-NEXT:                "castKind": "LValueToRValue",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 95
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 95
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "desugaredQualType": "int *",
// CHECK-NEXT:                   "qualType": "int *"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "__begin1",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "desugaredQualType": "int *",
// CHECK-NEXT:                    "qualType": "int *"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 96
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 96
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 98
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 14,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 98
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 13,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 98
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 98
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 13,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 98
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "isUsed": true,
// CHECK-NEXT:        "name": "C",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "Container"
// CHECK-NEXT:        },
// CHECK-NEXT:        "init": "call",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXConstructExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 13,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 98
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 13,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 98
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "Container"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue"
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXForRangeStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 99
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 100
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {},
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 16,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 99
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__range1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "Container &"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "Container"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "C",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "Container"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 14,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 99
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__begin1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMemberCallExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "MemberExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 99
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 99
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "<bound member function type>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "name": "begin",
// CHECK-NEXT:              "isArrow": false,
// CHECK-NEXT:              "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ImplicitCastExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 99
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 99
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "const Container"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "lvalue",
// CHECK-NEXT:                "castKind": "NoOp",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "Container"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "__range1",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "qualType": "Container &"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 14,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 99
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__end1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMemberCallExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "MemberExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 99
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 99
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "<bound member function type>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "name": "end",
// CHECK-NEXT:              "isArrow": false,
// CHECK-NEXT:              "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ImplicitCastExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 99
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 99
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "const Container"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "lvalue",
// CHECK-NEXT:                "castKind": "NoOp",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "Container"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "__range1",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "qualType": "Container &"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "BinaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "bool"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "opcode": "!=",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "int *",
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "__begin1",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "int *",
// CHECK-NEXT:              "qualType": "int *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "int *",
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "__end1",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "int *",
// CHECK-NEXT:              "qualType": "int *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "UnaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "desugaredQualType": "int *",
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "isPostfix": false,
// CHECK-NEXT:        "opcode": "++",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "__begin1",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "desugaredQualType": "int *",
// CHECK-NEXT:            "qualType": "int *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 8,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 99
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 12,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 99
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 99
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "name": "v",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 99
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "UnaryOperator",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 99
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 99
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "isPostfix": false,
// CHECK-NEXT:              "opcode": "*",
// CHECK-NEXT:              "canOverflow": false,
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ImplicitCastExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 99
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 99
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "desugaredQualType": "int *",
// CHECK-NEXT:                 "qualType": "int *"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "rvalue",
// CHECK-NEXT:                "castKind": "LValueToRValue",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "desugaredQualType": "int *",
// CHECK-NEXT:                   "qualType": "int *"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "__begin1",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "desugaredQualType": "int *",
// CHECK-NEXT:                    "qualType": "int *"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 100
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 100
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXForRangeStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 102
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 103
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 8,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 13,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 12,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 102
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 12,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "name": "a",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 23,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 23,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 23,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 102
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 23,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 23,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__range1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int (&)[10]"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 23,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 23,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int [10]"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "vals",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "int [10]"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 21,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 21,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 21,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 102
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__begin1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "ArrayToPointerDecay",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 21,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 102
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 21,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 102
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int [10]"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "VarDecl",
// CHECK-NEXT:               "name": "__range1",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int (&)[10]"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 21,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 21,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 21,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 102
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 23,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "isUsed": true,
// CHECK-NEXT:          "name": "__end1",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "BinaryOperator",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 23,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "opcode": "+",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "ImplicitCastExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 21,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 102
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 21,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 102
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int *"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "castKind": "ArrayToPointerDecay",
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "DeclRefExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 21,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 102
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 21,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 102
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "int [10]"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "lvalue",
// CHECK-NEXT:                "referencedDecl": {
// CHECK-NEXT:                 "id": "0x{{.*}}",
// CHECK-NEXT:                 "kind": "VarDecl",
// CHECK-NEXT:                 "name": "__range1",
// CHECK-NEXT:                 "type": {
// CHECK-NEXT:                  "qualType": "int (&)[10]"
// CHECK-NEXT:                 }
// CHECK-NEXT:                }
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "IntegerLiteral",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 23,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 102
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 23,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 102
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "long"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "value": "10"
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "BinaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 21,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 21,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "bool"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "opcode": "!=",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "int *",
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "__begin1",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "int *",
// CHECK-NEXT:              "qualType": "int *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "int *",
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "__end1",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "int *",
// CHECK-NEXT:              "qualType": "int *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "UnaryOperator",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 21,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 21,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "desugaredQualType": "int *",
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "isPostfix": false,
// CHECK-NEXT:        "opcode": "++",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "int *",
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "__begin1",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "desugaredQualType": "int *",
// CHECK-NEXT:            "qualType": "int *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 27,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 102
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 19,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 102
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 15,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 21,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 102
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "name": "v",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "init": "c",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 21,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 102
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "UnaryOperator",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 21,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 102
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 21,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 102
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "isPostfix": false,
// CHECK-NEXT:              "opcode": "*",
// CHECK-NEXT:              "canOverflow": false,
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ImplicitCastExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 21,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 102
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 21,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 102
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "desugaredQualType": "int *",
// CHECK-NEXT:                 "qualType": "int *"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "rvalue",
// CHECK-NEXT:                "castKind": "LValueToRValue",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 21,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 102
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 21,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 102
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "desugaredQualType": "int *",
// CHECK-NEXT:                   "qualType": "int *"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "__begin1",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "desugaredQualType": "int *",
// CHECK-NEXT:                    "qualType": "int *"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "NullStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 103
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 103
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }

