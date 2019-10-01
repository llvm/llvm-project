// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -ast-dump=json %s | FileCheck %s

struct A;
struct B;

struct A {
  int a;
  int b, c;
  int d : 12;
  int : 0;
  int e : 10;
  B *f;
};

struct C {
  struct {
    int a;
  } b;

  union {
    int c;
    float d;
  };

  struct {
    int e, f;
  };
};

struct D {
  int a;
  int b[10];
  int c[];
};

union E;

union F;

union E {
  int a;
  int b, c;
  int d : 12;
  int : 0;
  int e : 10;
  B *f;
};

union G {
  struct {
    int a;
  } b;

  union {
    int c;
    float d;
  };

  struct {
    int e, f;
  };
};

struct Base1 {};
struct Base2 {};
struct Base3 {};

struct Derived1 : Base1 {
};

struct Derived2 : private Base1 {
};

struct Derived3 : virtual Base1 {
};

struct Derived4 : Base1, virtual Base2, protected Base3 {
};

struct Derived5 : protected virtual Base1 {
};

template <typename... Bases>
struct Derived6 : virtual public Bases... {
};


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 3
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 3
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 8, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 3
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "A", 
// CHECK-NEXT:  "tagUsed": "struct"
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
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
// CHECK-NEXT:    "col": 8, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 4
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "isReferenced": true, 
// CHECK-NEXT:  "name": "B", 
// CHECK-NEXT:  "tagUsed": "struct"
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 6
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 6
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 13
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "previousDecl": "0x{{.*}}", 
// CHECK-NEXT:  "name": "A", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isPOD": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 6
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 6
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 6
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "A", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 7
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 7
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 7
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "a", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 8
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 8
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 8
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "b", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 10, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 8
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 8
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 10, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 8
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "c", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 9
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 9
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 9
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "d", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isBitfield": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ConstantExpr", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 9
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 9
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "valueCategory": "rvalue", 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "IntegerLiteral", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 11, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 9
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 9
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "value": "12"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 10
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 10
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 10
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isBitfield": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ConstantExpr", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 10
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 10
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "valueCategory": "rvalue", 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "IntegerLiteral", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 9, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 10
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 10
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "value": "0"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 11
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 11
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 11
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "e", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isBitfield": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ConstantExpr", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 11
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 11
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "valueCategory": "rvalue", 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "IntegerLiteral", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 11, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 11
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 11
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "value": "10"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 12
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 12
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 6, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 12
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "f", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "B *"
// CHECK-NEXT:    }
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 15
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 15
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 28
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "C", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasVariantMembers": true, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isPOD": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 15
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 15
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 15
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "C", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 16
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 16
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 18
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "tagUsed": "struct", 
// CHECK-NEXT:    "completeDefinition": true, 
// CHECK-NEXT:    "definitionData": {
// CHECK-NEXT:     "canPassInRegisters": true, 
// CHECK-NEXT:     "copyAssign": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "copyCtor": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "defaultCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "dtor": {
// CHECK-NEXT:      "irrelevant": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "isAggregate": true, 
// CHECK-NEXT:     "isLiteral": true, 
// CHECK-NEXT:     "isPOD": true, 
// CHECK-NEXT:     "isStandardLayout": true, 
// CHECK-NEXT:     "isTrivial": true, 
// CHECK-NEXT:     "isTriviallyCopyable": true, 
// CHECK-NEXT:     "moveAssign": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "moveCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 9, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 17
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 17
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 17
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "a", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 5, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 18
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 16
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 5, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 18
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "b", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "desugaredQualType": "C::(anonymous struct at {{.*}}:16:3)", 
// CHECK-NEXT:     "qualType": "struct (anonymous struct at {{.*}}:16:3)"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 20
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 20
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 23
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "tagUsed": "union", 
// CHECK-NEXT:    "completeDefinition": true, 
// CHECK-NEXT:    "definitionData": {
// CHECK-NEXT:     "canPassInRegisters": true, 
// CHECK-NEXT:     "copyAssign": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "copyCtor": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "defaultCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "dtor": {
// CHECK-NEXT:      "irrelevant": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "hasVariantMembers": true, 
// CHECK-NEXT:     "isAggregate": true, 
// CHECK-NEXT:     "isLiteral": true, 
// CHECK-NEXT:     "isPOD": true, 
// CHECK-NEXT:     "isStandardLayout": true, 
// CHECK-NEXT:     "isTrivial": true, 
// CHECK-NEXT:     "isTriviallyCopyable": true, 
// CHECK-NEXT:     "moveAssign": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "moveCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 9, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 21
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 21
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 21
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "c", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }, 
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 11, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 22
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 22
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 22
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "d", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "float"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 20
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 20
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 20
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "C::(anonymous union at {{.*}}:20:3)"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "IndirectFieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 9, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 21
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 21
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 21
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "c"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "IndirectFieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 11, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 22
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 22
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 22
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "d"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 25
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 25
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 27
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "tagUsed": "struct", 
// CHECK-NEXT:    "completeDefinition": true, 
// CHECK-NEXT:    "definitionData": {
// CHECK-NEXT:     "canPassInRegisters": true, 
// CHECK-NEXT:     "copyAssign": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "copyCtor": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "defaultCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "dtor": {
// CHECK-NEXT:      "irrelevant": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "isAggregate": true, 
// CHECK-NEXT:     "isLiteral": true, 
// CHECK-NEXT:     "isPOD": true, 
// CHECK-NEXT:     "isStandardLayout": true, 
// CHECK-NEXT:     "isTrivial": true, 
// CHECK-NEXT:     "isTriviallyCopyable": true, 
// CHECK-NEXT:     "moveAssign": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "moveCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 9, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 26
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 26
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 26
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "e", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }, 
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 12, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 26
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 26
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 12, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 26
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "f", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 25
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 25
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 25
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "C::(anonymous struct at {{.*}}:25:3)"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "IndirectFieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 9, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 26
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 26
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 26
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "e"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "IndirectFieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 12, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 26
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 12, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 26
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 12, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 26
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "f"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 30
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 30
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 34
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "D", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isPOD": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 30
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 30
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 30
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "D", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 31
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 31
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 31
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "a", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 32
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 32
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 32
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "b", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int [10]"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 33
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 33
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 33
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "c", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int []"
// CHECK-NEXT:    }
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 7, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 36
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 36
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 7, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 36
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "E", 
// CHECK-NEXT:  "tagUsed": "union"
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 7, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 38
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 38
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 7, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 38
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "F", 
// CHECK-NEXT:  "tagUsed": "union"
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 7, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 40
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 40
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 47
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "previousDecl": "0x{{.*}}", 
// CHECK-NEXT:  "name": "E", 
// CHECK-NEXT:  "tagUsed": "union", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasVariantMembers": true, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isPOD": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
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
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 40
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "E", 
// CHECK-NEXT:    "tagUsed": "union"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 41
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 41
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 41
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "a", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 42
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 42
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 42
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "b", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 10, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 42
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 42
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 10, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 42
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "c", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 43
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 43
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 43
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "d", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isBitfield": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ConstantExpr", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 43
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 43
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "valueCategory": "rvalue", 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "IntegerLiteral", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 11, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 43
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 43
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "value": "12"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 44
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 44
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 44
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isBitfield": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ConstantExpr", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 44
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 44
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "valueCategory": "rvalue", 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "IntegerLiteral", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 9, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 44
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 44
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "value": "0"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 45
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 45
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 45
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "e", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isBitfield": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ConstantExpr", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 45
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 45
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "valueCategory": "rvalue", 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "IntegerLiteral", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 11, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 45
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 45
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "value": "10"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 46
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 6, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "f", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "B *"
// CHECK-NEXT:    }
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 7, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 49
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 49
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 62
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "G", 
// CHECK-NEXT:  "tagUsed": "union", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasVariantMembers": true, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isPOD": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 49
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 49
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 49
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "G", 
// CHECK-NEXT:    "tagUsed": "union"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 50
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 50
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 52
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "tagUsed": "struct", 
// CHECK-NEXT:    "completeDefinition": true, 
// CHECK-NEXT:    "definitionData": {
// CHECK-NEXT:     "canPassInRegisters": true, 
// CHECK-NEXT:     "copyAssign": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "copyCtor": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "defaultCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "dtor": {
// CHECK-NEXT:      "irrelevant": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "isAggregate": true, 
// CHECK-NEXT:     "isLiteral": true, 
// CHECK-NEXT:     "isPOD": true, 
// CHECK-NEXT:     "isStandardLayout": true, 
// CHECK-NEXT:     "isTrivial": true, 
// CHECK-NEXT:     "isTriviallyCopyable": true, 
// CHECK-NEXT:     "moveAssign": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "moveCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 9, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 51
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 51
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 51
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "a", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 5, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 52
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 50
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 5, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 52
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "b", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "desugaredQualType": "G::(anonymous struct at {{.*}}:50:3)", 
// CHECK-NEXT:     "qualType": "struct (anonymous struct at {{.*}}:50:3)"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 54
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 54
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 57
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "tagUsed": "union", 
// CHECK-NEXT:    "completeDefinition": true, 
// CHECK-NEXT:    "definitionData": {
// CHECK-NEXT:     "canPassInRegisters": true, 
// CHECK-NEXT:     "copyAssign": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "copyCtor": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "defaultCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "dtor": {
// CHECK-NEXT:      "irrelevant": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "hasVariantMembers": true, 
// CHECK-NEXT:     "isAggregate": true, 
// CHECK-NEXT:     "isLiteral": true, 
// CHECK-NEXT:     "isPOD": true, 
// CHECK-NEXT:     "isStandardLayout": true, 
// CHECK-NEXT:     "isTrivial": true, 
// CHECK-NEXT:     "isTriviallyCopyable": true, 
// CHECK-NEXT:     "moveAssign": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "moveCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 9, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 55
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 55
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 55
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "c", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }, 
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 11, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 56
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 56
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 56
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "d", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "float"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 54
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 54
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 54
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "G::(anonymous union at {{.*}}:54:3)"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "IndirectFieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 9, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 55
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 55
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 55
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "c"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "IndirectFieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 11, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 56
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 56
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 56
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "d"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 59
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 59
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 61
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "tagUsed": "struct", 
// CHECK-NEXT:    "completeDefinition": true, 
// CHECK-NEXT:    "definitionData": {
// CHECK-NEXT:     "canPassInRegisters": true, 
// CHECK-NEXT:     "copyAssign": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "copyCtor": {
// CHECK-NEXT:      "hasConstParam": true, 
// CHECK-NEXT:      "implicitHasConstParam": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "defaultCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "dtor": {
// CHECK-NEXT:      "irrelevant": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "isAggregate": true, 
// CHECK-NEXT:     "isLiteral": true, 
// CHECK-NEXT:     "isPOD": true, 
// CHECK-NEXT:     "isStandardLayout": true, 
// CHECK-NEXT:     "isTrivial": true, 
// CHECK-NEXT:     "isTriviallyCopyable": true, 
// CHECK-NEXT:     "moveAssign": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "moveCtor": {
// CHECK-NEXT:      "exists": true, 
// CHECK-NEXT:      "needsImplicit": true, 
// CHECK-NEXT:      "simple": true, 
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 9, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 60
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 60
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 60
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "e", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }, 
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "FieldDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 12, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 60
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 60
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 12, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 60
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "f", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "FieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 3, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 59
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 59
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 59
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "G::(anonymous struct at {{.*}}:59:3)"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "IndirectFieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 9, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 60
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 60
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 9, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 60
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "e"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "IndirectFieldDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 12, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 60
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 12, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 60
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 12, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 60
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "f"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 64
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 64
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 15, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 64
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "isReferenced": true, 
// CHECK-NEXT:  "name": "Base1", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "defaultedIsConstexpr": true, 
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "isConstexpr": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasConstexprNonCopyMoveConstructor": true, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isEmpty": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isPOD": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 64
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Base1", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 64
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base1 &(const Base1 &)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "constexpr": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 64
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 64
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 64
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const Base1 &"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 64
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base1 &(Base1 &&)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "constexpr": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 64
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 64
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 64
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "Base1 &&"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXDestructorDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 64
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "~Base1", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 65
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 65
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 15, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 65
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "isReferenced": true, 
// CHECK-NEXT:  "name": "Base2", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "defaultedIsConstexpr": true, 
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "isConstexpr": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasConstexprNonCopyMoveConstructor": true, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isEmpty": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isPOD": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 65
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Base2", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 65
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base2 &(const Base2 &)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "constexpr": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 65
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 65
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 65
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const Base2 &"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 65
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base2 &(Base2 &&)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "constexpr": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 65
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 65
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 65
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "Base2 &&"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXDestructorDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 65
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 65
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "~Base2", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 66
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 66
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 15, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 66
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "isReferenced": true, 
// CHECK-NEXT:  "name": "Base3", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "defaultedIsConstexpr": true, 
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "isConstexpr": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasConstexprNonCopyMoveConstructor": true, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isEmpty": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isPOD": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 66
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 66
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 66
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Base3", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 66
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 66
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 66
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base3 &(const Base3 &)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "constexpr": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 66
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 66
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 66
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const Base3 &"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 66
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 66
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 66
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base3 &(Base3 &&)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "constexpr": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 66
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 66
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 66
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "Base3 &&"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXDestructorDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 66
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 66
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 66
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "~Base3", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 68
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 68
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 69
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "Derived1", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "defaultedIsConstexpr": true, 
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "isConstexpr": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasConstexprNonCopyMoveConstructor": true, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isEmpty": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "bases": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "access": "public", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base1"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "writtenAccess": "none"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 68
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 68
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 68
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Derived1", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 71
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 71
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 72
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "Derived2", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "canPassInRegisters": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "defaultedIsConstexpr": true, 
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "isConstexpr": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasConstexprNonCopyMoveConstructor": true, 
// CHECK-NEXT:   "isEmpty": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "bases": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "access": "private", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base1"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "writtenAccess": "private"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 71
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 71
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 71
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Derived2", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 74
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 74
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 75
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "Derived3", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "nonTrivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "bases": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "access": "public", 
// CHECK-NEXT:    "isVirtual": true, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base1"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "writtenAccess": "none"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 74
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Derived3", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 74
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Derived3 &(const Derived3 &)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 74
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 74
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 74
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const Derived3 &"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 74
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Derived3 &(Derived3 &&)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 74
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 74
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 74
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "Derived3 &&"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXDestructorDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 74
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "~Derived3", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 77
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 77
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 78
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "Derived4", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "nonTrivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "bases": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "access": "public", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base1"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "writtenAccess": "none"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "access": "public", 
// CHECK-NEXT:    "isVirtual": true, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base2"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "writtenAccess": "none"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "access": "protected", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base3"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "writtenAccess": "protected"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 77
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Derived4", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 77
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Derived4 &(const Derived4 &)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 77
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 77
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 77
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const Derived4 &"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 77
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Derived4 &(Derived4 &&)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 77
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 77
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 77
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "Derived4 &&"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXDestructorDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 77
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "~Derived4", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 80
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 80
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 81
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "Derived5", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "nonTrivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "nonTrivial": true, 
// CHECK-NEXT:    "simple": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "bases": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "access": "protected", 
// CHECK-NEXT:    "isVirtual": true, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Base1"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "writtenAccess": "protected"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 80
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 80
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 80
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Derived5", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 80
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 80
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 80
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Derived5 &(const Derived5 &)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 80
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 80
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 80
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const Derived5 &"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 80
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 80
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 80
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "operator=", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Derived5 &(Derived5 &&)"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default", 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 80
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 80
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 80
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "Derived5 &&"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXDestructorDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 80
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 80
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 80
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "~Derived5", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inline": true, 
// CHECK-NEXT:    "explicitlyDefaulted": "default"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "CXXRecordDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 8, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 84
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 84
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 85
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "Derived6", 
// CHECK-NEXT:  "tagUsed": "struct", 
// CHECK-NEXT:  "completeDefinition": true, 
// CHECK-NEXT:  "definitionData": {
// CHECK-NEXT:   "canConstDefaultInit": true, 
// CHECK-NEXT:   "copyAssign": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "copyCtor": {
// CHECK-NEXT:    "hasConstParam": true, 
// CHECK-NEXT:    "implicitHasConstParam": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "defaultCtor": {
// CHECK-NEXT:    "defaultedIsConstexpr": true, 
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "isConstexpr": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "dtor": {
// CHECK-NEXT:    "irrelevant": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "hasConstexprNonCopyMoveConstructor": true, 
// CHECK-NEXT:   "isAggregate": true, 
// CHECK-NEXT:   "isEmpty": true, 
// CHECK-NEXT:   "isLiteral": true, 
// CHECK-NEXT:   "isStandardLayout": true, 
// CHECK-NEXT:   "isTrivial": true, 
// CHECK-NEXT:   "isTriviallyCopyable": true, 
// CHECK-NEXT:   "moveAssign": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "moveCtor": {
// CHECK-NEXT:    "exists": true, 
// CHECK-NEXT:    "needsImplicit": true, 
// CHECK-NEXT:    "simple": true, 
// CHECK-NEXT:    "trivial": true
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "bases": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "access": "public", 
// CHECK-NEXT:    "isPackExpansion": true, 
// CHECK-NEXT:    "isVirtual": true, 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "Bases"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "writtenAccess": "public"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CXXRecordDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 8, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 84
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 84
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 84
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "Derived6", 
// CHECK-NEXT:    "tagUsed": "struct"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }
