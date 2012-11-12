// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -fno-limit-debug-info %s -o - | FileCheck %s

class Test
{
public:
    Test () : reserved (new data()) {}

    unsigned
    getID() const
    {
        return reserved->objectID;
    }
protected:
    struct data {
        unsigned objectID;
    };
    data* reserved;
};

Test t;

// CHECK: metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata {{.*}} [ DW_TAG_pointer_type ]
// CHECK: metadata !"data", metadata !6, i32 14, i64 32, i64 32, i32 0, i32 0
// CHECK-NOT: metadata !"data", metadata {{.*}}, i32 14, i64 0, i64 0, i32 0, i32 4,
