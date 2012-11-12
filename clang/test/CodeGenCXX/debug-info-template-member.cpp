// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

class MyClass
{
public:
    int add2(int j)
    {
        return add<2>(j);
    }
private:
    template <int i> int add(int j)
    {
        return i + j;
    }
};

MyClass m;

// CHECK: metadata !{i32 {{.*}}, null, metadata !"MyClass", metadata {{.*}}, i32 {{.*}}, i64 8, i64 8, i32 0, i32 0, null, metadata [[C_MEM:.*]], i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK: [[C_MEM]] = metadata !{metadata {{.*}}, metadata [[C_TEMP:.*]], metadata {{.*}}}
// CHECK: [[C_TEMP]] = metadata !{i32 {{.*}}, i32 0, metadata {{.*}}, metadata !"add<2>", metadata !"add<2>", metadata !"_ZN7MyClass3addILi2EEEii", metadata {{.*}}
