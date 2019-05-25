// RUN: %clang_cc1 -Wno-unused -fblocks -ast-dump=json -ast-dump-filter Test %s | FileCheck %s

@protocol P
@end

@interface A
@end

@interface TestObjCIvarDecl : A
@end

@implementation TestObjCIvarDecl {
  int varDefault;
  @private int varPrivate;
  @protected int varProtected;
  @public int varPublic;
  @package int varPackage;
}
@end

@interface testObjCMethodDecl : A {
}
- (int) TestObjCMethodDecl: (int)i, ...;
@end

@implementation testObjCMethodDecl
- (int) TestObjCMethodDecl: (int)i, ... {
  return 0;
}
@end

@protocol TestObjCProtocolDecl
- (void) foo;
@end

@interface TestObjCClass : A <P>
- (void) foo;
@end

@implementation TestObjCClass : A {
  int i;
}
- (void) foo {
}
@end

@interface TestObjCClass (TestObjCCategoryDecl) <P>
- (void) bar;
@end

@interface TestGenericInterface<T> : A<P> {
}
@end

@implementation TestObjCClass (TestObjCCategoryDecl)
- (void) bar {
}
@end

@compatibility_alias TestObjCCompatibleAliasDecl A;

@interface TestObjCProperty: A
@property(getter=getterFoo, setter=setterFoo:) int foo;
@property int bar;
@end

@implementation TestObjCProperty {
  int i;
}
@synthesize foo=i;
@synthesize bar;
@end

void TestBlockDecl(int x) {
  ^(int y, ...){ x; };
}

@interface B
+ (int) foo;
@end

void f() {
  __typeof__(B.foo) Test;
}

// CHECK:  "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 12, 
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
// CHECK-NEXT:    "col": 2, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 10
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCIvarDecl", 
// CHECK-NEXT:  "super": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "A"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "implementation": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCImplementationDecl", 
// CHECK-NEXT:   "name": "TestObjCIvarDecl"
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCImplementationDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 17, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 12
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 12
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 19
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCIvarDecl", 
// CHECK-NEXT:  "super": {
// CHECK-NEXT:   "id": "0x{{.*}}"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "interface": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "TestObjCIvarDecl"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCIvarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 13
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 13
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 13
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "varDefault", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "access": "private"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCIvarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 16, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 14
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 12, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 14
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 16, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 14
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "varPrivate", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "access": "private"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCIvarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 18, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 15
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 14, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 15
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 18, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 15
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "varProtected", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "access": "protected"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCIvarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 15, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 16
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 11, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 16
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 15, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 16
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "varPublic", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "access": "public"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCIvarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 16, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 17
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 12, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 17
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 16, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 17
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "varPackage", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "access": "package"
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCMethodDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 1, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 23
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 23
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 40, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 23
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCMethodDecl:", 
// CHECK-NEXT:  "returnType": {
// CHECK-NEXT:   "qualType": "int"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "instance": true, 
// CHECK-NEXT:  "variadic": true, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ParmVarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 34, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 23
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 30, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 23
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 34, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 23
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "i", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCMethodDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 1, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 27
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 27
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 29
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCMethodDecl:", 
// CHECK-NEXT:  "returnType": {
// CHECK-NEXT:   "qualType": "int"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "instance": true, 
// CHECK-NEXT:  "variadic": true, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ImplicitParamDecl", 
// CHECK-NEXT:    "loc": {}, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {}, 
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "self", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "testObjCMethodDecl *"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ImplicitParamDecl", 
// CHECK-NEXT:    "loc": {}, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {}, 
// CHECK-NEXT:     "end": {}
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "_cmd", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "desugaredQualType": "SEL *", 
// CHECK-NEXT:     "qualType": "SEL"
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ParmVarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 34, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 27
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 30, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 27
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 34, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 27
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
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
// CHECK-NEXT:      "col": 41, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 27
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 29
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ReturnStmt", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 28
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 10, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 28
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}", 
// CHECK-NEXT:        "kind": "IntegerLiteral", 
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 10, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 28
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 10, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 28
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
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCProtocolDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 11, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 32
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 32
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 2, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 34
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCProtocolDecl", 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 1, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 33
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 33
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 13, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 33
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "foo", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "void"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 12, 
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
// CHECK-NEXT:    "col": 2, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 38
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCClass", 
// CHECK-NEXT:  "super": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "A"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "implementation": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCImplementationDecl", 
// CHECK-NEXT:   "name": "TestObjCClass"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "protocols": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCProtocolDecl", 
// CHECK-NEXT:    "name": "P"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 1, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 37
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 37
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 13, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 37
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "foo", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "void"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCImplementationDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 17, 
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
// CHECK-NEXT:    "line": 45
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCClass", 
// CHECK-NEXT:  "super": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "A"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "interface": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "TestObjCClass"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCIvarDecl", 
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
// CHECK-NEXT:    "name": "i", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "access": "private"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 1, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 43
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 43
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 44
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "foo", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "void"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ImplicitParamDecl", 
// CHECK-NEXT:      "loc": {}, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {}, 
// CHECK-NEXT:       "end": {}
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "isImplicit": true, 
// CHECK-NEXT:      "name": "self", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "TestObjCClass *"
// CHECK-NEXT:      }
// CHECK-NEXT:     }, 
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ImplicitParamDecl", 
// CHECK-NEXT:      "loc": {}, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {}, 
// CHECK-NEXT:       "end": {}
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "isImplicit": true, 
// CHECK-NEXT:      "name": "_cmd", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "desugaredQualType": "SEL *", 
// CHECK-NEXT:       "qualType": "SEL"
// CHECK-NEXT:      }
// CHECK-NEXT:     }, 
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "CompoundStmt", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 14, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 43
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 1, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 44
// CHECK-NEXT:       }
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCCategoryDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 12, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 47
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 47
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 2, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 49
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCCategoryDecl", 
// CHECK-NEXT:  "interface": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "TestObjCClass"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "implementation": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCCategoryImplDecl", 
// CHECK-NEXT:   "name": "TestObjCCategoryDecl"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "protocols": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCProtocolDecl", 
// CHECK-NEXT:    "name": "P"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 1, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 48
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 48
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 13, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 48
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "bar", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "void"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 12, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 51
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 51
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 2, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 53
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestGenericInterface", 
// CHECK-NEXT:  "super": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "A"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "implementation": {
// CHECK-NEXT:   "id": "0x{{.*}}"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "protocols": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCProtocolDecl", 
// CHECK-NEXT:    "name": "P"
// CHECK-NEXT:   }
// CHECK-NEXT:  ], 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCTypeParamDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 33, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 51
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 33, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 51
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 33, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 51
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "T", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "desugaredQualType": "id", 
// CHECK-NEXT:     "qualType": "id"
// CHECK-NEXT:    }
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCCategoryImplDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 17, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 55
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 55
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 58
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCCategoryDecl", 
// CHECK-NEXT:  "interface": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "TestObjCClass"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "categoryDecl": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCCategoryDecl", 
// CHECK-NEXT:   "name": "TestObjCCategoryDecl"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 1, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 56
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 56
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 57
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "bar", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "void"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ImplicitParamDecl", 
// CHECK-NEXT:      "loc": {}, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {}, 
// CHECK-NEXT:       "end": {}
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "isImplicit": true, 
// CHECK-NEXT:      "name": "self", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "TestObjCClass *"
// CHECK-NEXT:      }
// CHECK-NEXT:     }, 
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ImplicitParamDecl", 
// CHECK-NEXT:      "loc": {}, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {}, 
// CHECK-NEXT:       "end": {}
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "isImplicit": true, 
// CHECK-NEXT:      "name": "_cmd", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "desugaredQualType": "SEL *", 
// CHECK-NEXT:       "qualType": "SEL"
// CHECK-NEXT:      }
// CHECK-NEXT:     }, 
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "CompoundStmt", 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 14, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 56
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 1, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 57
// CHECK-NEXT:       }
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCCompatibleAliasDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 1, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 60
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 60
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 60
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCCompatibleAliasDecl", 
// CHECK-NEXT:  "interface": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "A"
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 12, 
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
// CHECK-NEXT:    "col": 2, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 65
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCProperty", 
// CHECK-NEXT:  "super": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "A"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "implementation": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCImplementationDecl", 
// CHECK-NEXT:   "name": "TestObjCProperty"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCPropertyDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 52, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 63
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 63
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 52, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 63
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "foo", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "getter": {
// CHECK-NEXT:     "id": "0x{{.*}}", 
// CHECK-NEXT:     "kind": "ObjCMethodDecl", 
// CHECK-NEXT:     "name": "getterFoo"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "setter": {
// CHECK-NEXT:     "id": "0x{{.*}}", 
// CHECK-NEXT:     "kind": "ObjCMethodDecl", 
// CHECK-NEXT:     "name": "setterFoo:"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "assign": true, 
// CHECK-NEXT:    "readwrite": true, 
// CHECK-NEXT:    "atomic": true, 
// CHECK-NEXT:    "unsafe_unretained": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCPropertyDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 15, 
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
// CHECK-NEXT:      "col": 15, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "bar", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "assign": true, 
// CHECK-NEXT:    "readwrite": true, 
// CHECK-NEXT:    "atomic": true, 
// CHECK-NEXT:    "unsafe_unretained": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 52, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 63
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 52, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 63
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 52, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 63
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "getterFoo", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 52, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 63
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 52, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 63
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 52, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 63
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "setterFoo:", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "void"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 52, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 63
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 52, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 63
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 52, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 63
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "foo", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 15, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 64
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 15, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 15, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "bar", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCMethodDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 15, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 64
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 15, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 15, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 64
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "isImplicit": true, 
// CHECK-NEXT:    "name": "setBar:", 
// CHECK-NEXT:    "returnType": {
// CHECK-NEXT:     "qualType": "void"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "instance": true, 
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}", 
// CHECK-NEXT:      "kind": "ParmVarDecl", 
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 15, 
// CHECK-NEXT:       "file": "{{.*}}", 
// CHECK-NEXT:       "line": 64
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 15, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 64
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 15, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 64
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "name": "bar", 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "ObjCImplementationDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 17, 
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
// CHECK-NEXT:    "line": 72
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestObjCProperty", 
// CHECK-NEXT:  "super": {
// CHECK-NEXT:   "id": "0x{{.*}}"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "interface": {
// CHECK-NEXT:   "id": "0x{{.*}}", 
// CHECK-NEXT:   "kind": "ObjCInterfaceDecl", 
// CHECK-NEXT:   "name": "TestObjCProperty"
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCIvarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 7, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 68
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 3, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 68
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 68
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "i", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "access": "private"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCPropertyImplDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 13, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 70
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 70
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 17, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 70
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "foo", 
// CHECK-NEXT:    "implKind": "synthesize", 
// CHECK-NEXT:    "propertyDecl": {
// CHECK-NEXT:     "id": "0x{{.*}}", 
// CHECK-NEXT:     "kind": "ObjCPropertyDecl", 
// CHECK-NEXT:     "name": "foo"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "ivarDecl": {
// CHECK-NEXT:     "id": "0x{{.*}}", 
// CHECK-NEXT:     "kind": "ObjCIvarDecl", 
// CHECK-NEXT:     "name": "i", 
// CHECK-NEXT:     "type": {
// CHECK-NEXT:      "qualType": "int"
// CHECK-NEXT:     }
// CHECK-NEXT:    }
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCIvarDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 13, 
// CHECK-NEXT:     "file": "{{.*}}", 
// CHECK-NEXT:     "line": 71
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 13, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 71
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 13, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 71
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "bar", 
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "synthesized": true, 
// CHECK-NEXT:    "access": "private"
// CHECK-NEXT:   }, 
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}", 
// CHECK-NEXT:    "kind": "ObjCPropertyImplDecl", 
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 13, 
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
// CHECK-NEXT:      "col": 13, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 71
// CHECK-NEXT:     }
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "name": "bar", 
// CHECK-NEXT:    "implKind": "synthesize", 
// CHECK-NEXT:    "propertyDecl": {
// CHECK-NEXT:     "id": "0x{{.*}}", 
// CHECK-NEXT:     "kind": "ObjCPropertyDecl", 
// CHECK-NEXT:     "name": "bar"
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "ivarDecl": {
// CHECK-NEXT:     "id": "0x{{.*}}", 
// CHECK-NEXT:     "kind": "ObjCIvarDecl", 
// CHECK-NEXT:     "name": "bar", 
// CHECK-NEXT:     "type": {
// CHECK-NEXT:      "qualType": "int"
// CHECK-NEXT:     }
// CHECK-NEXT:    }
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6, 
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
// CHECK-NEXT:    "line": 76
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "TestBlockDecl", 
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
// CHECK-NEXT:     "line": 74
// CHECK-NEXT:    }, 
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 20, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 24, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 74
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
// CHECK-NEXT:      "line": 74
// CHECK-NEXT:     }, 
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1, 
// CHECK-NEXT:      "file": "{{.*}}", 
// CHECK-NEXT:      "line": 76
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
// CHECK-NEXT:        "line": 75
// CHECK-NEXT:       }, 
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 21, 
// CHECK-NEXT:        "file": "{{.*}}", 
// CHECK-NEXT:        "line": 75
// CHECK-NEXT:       }
// CHECK-NEXT:      }, 
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void (^)(int, ...)"
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
// CHECK-NEXT:          "line": 75
// CHECK-NEXT:         }, 
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 21, 
// CHECK-NEXT:          "file": "{{.*}}", 
// CHECK-NEXT:          "line": 75
// CHECK-NEXT:         }
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (^)(int, ...)"
// CHECK-NEXT:        }, 
// CHECK-NEXT:        "valueCategory": "rvalue", 
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}", 
// CHECK-NEXT:          "kind": "BlockDecl", 
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3, 
// CHECK-NEXT:           "file": "{{.*}}", 
// CHECK-NEXT:           "line": 75
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 75
// CHECK-NEXT:           }, 
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 21, 
// CHECK-NEXT:            "file": "{{.*}}", 
// CHECK-NEXT:            "line": 75
// CHECK-NEXT:           }
// CHECK-NEXT:          }, 
// CHECK-NEXT:          "variadic": true, 
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}", 
// CHECK-NEXT:            "kind": "ParmVarDecl", 
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 9, 
// CHECK-NEXT:             "file": "{{.*}}", 
// CHECK-NEXT:             "line": 75
// CHECK-NEXT:            }, 
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 75
// CHECK-NEXT:             }, 
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 9, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 75
// CHECK-NEXT:             }
// CHECK-NEXT:            }, 
// CHECK-NEXT:            "name": "y", 
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            }
// CHECK-NEXT:           }, 
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
// CHECK-NEXT:              "col": 16, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 75
// CHECK-NEXT:             }, 
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 21, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 75
// CHECK-NEXT:             }
// CHECK-NEXT:            }, 
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}", 
// CHECK-NEXT:              "kind": "ImplicitCastExpr", 
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 18, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 75
// CHECK-NEXT:               }, 
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 18, 
// CHECK-NEXT:                "file": "{{.*}}", 
// CHECK-NEXT:                "line": 75
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
// CHECK-NEXT:                  "col": 18, 
// CHECK-NEXT:                  "file": "{{.*}}", 
// CHECK-NEXT:                  "line": 75
// CHECK-NEXT:                 }, 
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 18, 
// CHECK-NEXT:                  "file": "{{.*}}", 
// CHECK-NEXT:                  "line": 75
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
// CHECK-NEXT:           }, 
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}", 
// CHECK-NEXT:            "kind": "ParmVarDecl", 
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 9, 
// CHECK-NEXT:             "file": "{{.*}}", 
// CHECK-NEXT:             "line": 75
// CHECK-NEXT:            }, 
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 75
// CHECK-NEXT:             }, 
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 9, 
// CHECK-NEXT:              "file": "{{.*}}", 
// CHECK-NEXT:              "line": 75
// CHECK-NEXT:             }
// CHECK-NEXT:            }, 
// CHECK-NEXT:            "name": "y", 
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            }
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


// CHECK:  "kind": "VarDecl", 
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 21, 
// CHECK-NEXT:   "file": "{{.*}}", 
// CHECK-NEXT:   "line": 83
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 3, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 83
// CHECK-NEXT:   }, 
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 21, 
// CHECK-NEXT:    "file": "{{.*}}", 
// CHECK-NEXT:    "line": 83
// CHECK-NEXT:   }
// CHECK-NEXT:  }, 
// CHECK-NEXT:  "name": "Test", 
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "desugaredQualType": "int", 
// CHECK-NEXT:   "qualType": "typeof (B.foo)"
// CHECK-NEXT:  }
// CHECK-NEXT: }

