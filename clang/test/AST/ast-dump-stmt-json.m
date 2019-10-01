// RUN: %clang_cc1 -triple x86_64-pc-win32 -Wno-unused -fblocks -fobjc-exceptions -ast-dump=json -ast-dump-filter Test %s | FileCheck %s

void TestBlockExpr(int x) {
  ^{ x; };
}

void TestExprWithCleanup(int x) {
  ^{ x; };
}

@interface A
@end

void TestObjCAtCatchStmt() {
  @try {
  } @catch(A *a) {
  } @catch(...) {
  } @finally {
  }
}


// CHECK:  "kind": "FunctionDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6, 
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
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 5
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestBlockExpr", 
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (int)"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ParmVarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 24, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 3
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 20, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 3
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 24, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 3
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isUsed": true, 
// CHECK-NEXT:    "name": "x", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CompoundStmt", 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 27, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 3
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 5
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ExprWithCleanups", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 4
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 4
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void (^)(void)"
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "valueCategory": "rvalue", 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "BlockExpr", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 4
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 4
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (^)(void)"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}", 
// CHECK-NEXT:          "kind": "BlockDecl", 
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3, 
// CHECK-NEXT:           "file": "{{.*}}", 
// CHECK-NEXT:           "line": 4
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 4
// CHECK-NEXT:           }, 
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 9, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 4
// CHECK-NEXT:           }
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "kind": "Capture", 
// CHECK-NEXT:            "var": {
// CHECK-NEXT:             "id": "0x{{.*}}", 
// CHECK-NEXT:             "kind": "ParmVarDecl", 
// CHECK-NEXT:             "name": "x", 
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "int"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }, 
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}", 
// CHECK-NEXT:            "kind": "CompoundStmt", 
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 4
// CHECK-NEXT:             }, 
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 9, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 4
// CHECK-NEXT:             }
// CHECK-NEXT:            }, 
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}", 
// CHECK-NEXT:              "kind": "ImplicitCastExpr", 
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 6, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 4
// CHECK-NEXT:               }, 
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 6, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 4
// CHECK-NEXT:               }
// CHECK-NEXT:              }, 
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              }, 
// CHECK-NEXT:              "valueCategory": "rvalue", 
// CHECK-NEXT:              "castKind": "LValueToRValue", 
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}", 
// CHECK-NEXT:                "kind": "DeclRefExpr", 
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 6, 
// CHECK-NEXT:                  "file": "{{.*}}", 
// CHECK-NEXT:                  "line": 4
// CHECK-NEXT:                 }, 
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 6, 
// CHECK-NEXT:                  "file": "{{.*}}", 
// CHECK-NEXT:                  "line": 4
// CHECK-NEXT:                 }
// CHECK-NEXT:                }, 
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "const int"
// CHECK-NEXT:                }, 
// CHECK-NEXT:                "valueCategory": "lvalue", 
// CHECK-NEXT:                "referencedDecl": {
// CHECK-NEXT:                 "id": "0x{{.*}}", 
// CHECK-NEXT:                 "kind": "ParmVarDecl", 
// CHECK-NEXT:                 "name": "x", 
// CHECK-NEXT:                 "type": {
// CHECK-NEXT:                  "qualType": "int"
// CHECK-NEXT:                 }
// CHECK-NEXT:                }
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
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
// CHECK-NEXT:   "line": 7
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 7
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 9
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestExprWithCleanup", 
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (int)"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ParmVarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 30, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 7
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 26, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 7
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 30, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 7
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isUsed": true, 
// CHECK-NEXT:    "name": "x", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "CompoundStmt", 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 33, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 7
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 9
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ExprWithCleanups", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 8
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 8
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void (^)(void)"
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "valueCategory": "rvalue", 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "BlockExpr", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 8
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 8
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (^)(void)"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}", 
// CHECK-NEXT:          "kind": "BlockDecl", 
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3, 
// CHECK-NEXT:           "file": "{{.*}}", 
// CHECK-NEXT:           "line": 8
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 8
// CHECK-NEXT:           }, 
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 9, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 8
// CHECK-NEXT:           }
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "kind": "Capture", 
// CHECK-NEXT:            "var": {
// CHECK-NEXT:             "id": "0x{{.*}}", 
// CHECK-NEXT:             "kind": "ParmVarDecl", 
// CHECK-NEXT:             "name": "x", 
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "int"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }, 
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}", 
// CHECK-NEXT:            "kind": "CompoundStmt", 
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 8
// CHECK-NEXT:             }, 
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 9, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 8
// CHECK-NEXT:             }
// CHECK-NEXT:            }, 
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}", 
// CHECK-NEXT:              "kind": "ImplicitCastExpr", 
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 6, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 8
// CHECK-NEXT:               }, 
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 6, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 8
// CHECK-NEXT:               }
// CHECK-NEXT:              }, 
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              }, 
// CHECK-NEXT:              "valueCategory": "rvalue", 
// CHECK-NEXT:              "castKind": "LValueToRValue", 
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}", 
// CHECK-NEXT:                "kind": "DeclRefExpr", 
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 6, 
// CHECK-NEXT:                  "file": "{{.*}}", 
// CHECK-NEXT:                  "line": 8
// CHECK-NEXT:                 }, 
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 6, 
// CHECK-NEXT:                  "file": "{{.*}}", 
// CHECK-NEXT:                  "line": 8
// CHECK-NEXT:                 }
// CHECK-NEXT:                }, 
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "const int"
// CHECK-NEXT:                }, 
// CHECK-NEXT:                "valueCategory": "lvalue", 
// CHECK-NEXT:                "referencedDecl": {
// CHECK-NEXT:                 "id": "0x{{.*}}", 
// CHECK-NEXT:                 "kind": "ParmVarDecl", 
// CHECK-NEXT:                 "name": "x", 
// CHECK-NEXT:                 "type": {
// CHECK-NEXT:                  "qualType": "int"
// CHECK-NEXT:                 }
// CHECK-NEXT:                }
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
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
// CHECK-NEXT:    "line": 20
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCAtCatchStmt", 
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
// CHECK-NEXT:      "line": 14
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 20
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ObjCAtTryStmt", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 15
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 3, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "CompoundStmt", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 8, 
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
// CHECK-NEXT:        "kind": "ObjCAtCatchStmt", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 16
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 17
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}", 
// CHECK-NEXT:          "kind": "VarDecl", 
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 15, 
// CHECK-NEXT:           "file": "{{.*}}", 
// CHECK-NEXT:           "line": 16
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 12, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 16
// CHECK-NEXT:           }, 
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 15, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 16
// CHECK-NEXT:           }
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "name": "a", 
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "A *"
// CHECK-NEXT:          }
// CHECK-NEXT:         }, 
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}", 
// CHECK-NEXT:          "kind": "CompoundStmt", 
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 18, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 16
// CHECK-NEXT:           }, 
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 17
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }, 
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "ObjCAtCatchStmt", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 17
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 18
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "isCatchAll": true, 
// CHECK-NEXT:        "inner": [
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
// CHECK-NEXT:       }, 
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "ObjCAtFinallyStmt", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 18
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 19
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}", 
// CHECK-NEXT:          "kind": "CapturedStmt", 
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 18
// CHECK-NEXT:           }, 
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 19
// CHECK-NEXT:           }
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}", 
// CHECK-NEXT:            "kind": "CapturedDecl", 
// CHECK-NEXT:            "loc": {}, 
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {}, 
// CHECK-NEXT:             "end": {}
// CHECK-NEXT:            }, 
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}", 
// CHECK-NEXT:              "kind": "CompoundStmt", 
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 18
// CHECK-NEXT:               }, 
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 3, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 19
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }, 
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}", 
// CHECK-NEXT:              "kind": "ImplicitParamDecl", 
// CHECK-NEXT:              "loc": {
// CHECK-NEXT:               "col": 14, 
// CHECK-NEXT:               "file": "{{.*}}", 
// CHECK-NEXT:               "line": 18
// CHECK-NEXT:              }, 
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 18
// CHECK-NEXT:               }, 
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 14, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 18
// CHECK-NEXT:               }
// CHECK-NEXT:              }, 
// CHECK-NEXT:              "isImplicit": true, 
// CHECK-NEXT:              "name": "__context", 
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "struct (anonymous at {{.*}}:18:14) *"
// CHECK-NEXT:              }
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

