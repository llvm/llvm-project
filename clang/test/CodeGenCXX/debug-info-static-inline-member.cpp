// RUN: %clangxx -target arm64-apple-macosx11.0.0 -g %s -emit-llvm -S -o - | FileCheck --check-prefixes=CHECK %s

enum class Enum : int {
  VAL = -1
};

struct Empty {};

constexpr auto func() { return 25; }

struct Foo {
    static constexpr int cexpr_int = func();
    static constexpr int cexpr_int2 = func() + 1;
    static constexpr float cexpr_float = 2.0 + 1.0;
    static constexpr Enum cexpr_enum = Enum::VAL;
    static constexpr Empty cexpr_empty{};

    template<typename T>
    static constexpr T cexpr_template{};
};

int main() {
    Foo f;

    // Force global variable definitions to be emitted.
    (void)&Foo::cexpr_int;
    (void)&Foo::cexpr_empty;

    return Foo::cexpr_int + Foo::cexpr_float
           + (int)Foo::cexpr_enum + Foo::cexpr_template<short>;
}

// CHECK:      @{{.*}}cexpr_int{{.*}} =
// CHECK-SAME:   !dbg ![[INT_GLOBAL:[0-9]+]]

// CHECK:      @{{.*}}cexpr_empty{{.*}} = 
// CHECK-SAME    !dbg ![[EMPTY_GLOBAL:[0-9]+]]

// CHECK:      !DIGlobalVariableExpression(var: ![[INT_VAR:[0-9]+]], expr: !DIExpression())
// CHECK:      ![[INT_VAR]] = distinct !DIGlobalVariable(name: "cexpr_int", linkageName:
// CHECK-SAME:                isLocal: false, isDefinition: true, declaration: ![[INT_DECL:[0-9]+]])

// CHECK:      ![[INT_DECL]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_int",
// CHECK-SAME:                 flags: DIFlagStaticMember, extraData: i32 25)

// CHECK:      ![[INT_DECL2:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_int2",
// CHECK-SAME:                         flags: DIFlagStaticMember, extraData: i32 26)

// CHECK:      ![[FLOAT_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_float",
// CHECK-SAME:                          flags: DIFlagStaticMember, extraData: float

// CHECK:      ![[ENUM_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_enum",
// CHECK-SAME:                         flags: DIFlagStaticMember, extraData: i32 -1)

// CHECK:      ![[EMPTY_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_empty",
// CHECK-SAME:                          flags: DIFlagStaticMember)

// CHECK:      ![[TEMPLATE_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_template",
// CHECK-SAME:                             flags: DIFlagStaticMember, extraData: i16 0)

// CHECK:      !DIGlobalVariableExpression(var: ![[EMPTY_VAR:[0-9]+]], expr: !DIExpression())
// CHECK:      ![[EMPTY_VAR]] = distinct !DIGlobalVariable(name: "cexpr_empty", linkageName:
// CHECK-SAME:                  isLocal: false, isDefinition: true, declaration: ![[EMPTY_DECL]])

// CHECK:      !DIGlobalVariableExpression(var: ![[INT_VAR2:[0-9]+]], expr: !DIExpression(DW_OP_constu, 26, DW_OP_stack_value))
// CHECK:      ![[INT_VAR2]] = distinct !DIGlobalVariable(name: "cexpr_int2", linkageName:
// CHECK-SAME:                 isLocal: true, isDefinition: true, declaration: ![[INT_DECL2]])

// CHECK:      !DIGlobalVariableExpression(var: ![[FLOAT_VAR:[0-9]+]], expr: !DIExpression(DW_OP_constu, {{.*}}, DW_OP_stack_value))
// CHECK:      ![[FLOAT_VAR]] = distinct !DIGlobalVariable(name: "cexpr_float", linkageName:
// CHECK-SAME:                  isLocal: true, isDefinition: true, declaration: ![[FLOAT_DECL]])

// CHECK:      !DIGlobalVariableExpression(var: ![[ENUM_VAR:[0-9]+]], expr: !DIExpression(DW_OP_constu, {{.*}}, DW_OP_stack_value))
// CHECK:      ![[ENUM_VAR]] = distinct !DIGlobalVariable(name: "cexpr_enum", linkageName:
// CHECK-SAME:                 isLocal: true, isDefinition: true, declaration: ![[ENUM_DECL]])

// CHECK:      !DIGlobalVariableExpression(var: ![[TEMPLATE_VAR:[0-9]+]], expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
// CHECK:      ![[TEMPLATE_VAR]] = distinct !DIGlobalVariable(name: "cexpr_template", linkageName:
// CHECK-SAME:                     isLocal: true, isDefinition: true, declaration: ![[TEMPLATE_DECL]], templateParams: ![[TEMPLATE_PARMS:[0-9]+]])
