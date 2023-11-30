// RUN: %clangxx -target arm64-apple-macosx11.0.0 -g -gdwarf-4 -debug-info-kind=standalone %s -emit-llvm -S -o - | FileCheck --check-prefixes=CHECK %s
// RUN: %clangxx -target arm64-apple-macosx11.0.0 -g -gdwarf-4 -debug-info-kind=limited %s -emit-llvm -S -o - | FileCheck --check-prefixes=CHECK %s

enum class Enum : int {
  VAL = -1
};

struct Empty {};
struct Fwd;

constexpr auto func() { return 25; }

struct Foo {
    static constexpr int cexpr_int_with_addr = func();
    static constexpr int cexpr_int2 = func() + 1;
    static constexpr float cexpr_float = 2.0 + 1.0;
    static constexpr Enum cexpr_enum = Enum::VAL;
    static constexpr Empty cexpr_struct_with_addr{};
    static inline    Enum inline_enum = Enum::VAL;

    template<typename T, unsigned V>
    static constexpr auto cexpr_template = V;

    static const auto empty_templated = cexpr_template<Fwd, 1>;
};

int main() {
    Foo f;
    //Bar b;

    // Force global variable definitions to be emitted.
    (void)&Foo::cexpr_int_with_addr;
    (void)&Foo::cexpr_struct_with_addr;

    return Foo::cexpr_int_with_addr + Foo::cexpr_float
           + (int)Foo::cexpr_enum + Foo::cexpr_template<short, 5>
           + Foo::empty_templated;
}

// CHECK:      @{{.*}}cexpr_int_with_addr{{.*}} =
// CHECK-SAME:   !dbg ![[INT_GLOBAL:[0-9]+]]

// CHECK:      @{{.*}}cexpr_struct_with_addr{{.*}} = 
// CHECK-SAME    !dbg ![[EMPTY_GLOBAL:[0-9]+]]

// CHECK:      !DIGlobalVariableExpression(var: ![[INT_VAR:[0-9]+]], expr: !DIExpression())
// CHECK:      ![[INT_VAR]] = distinct !DIGlobalVariable(name: "cexpr_int_with_addr", linkageName:
// CHECK-SAME:                isLocal: false, isDefinition: true, declaration: ![[INT_DECL:[0-9]+]])

// CHECK:      ![[INT_DECL]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_int_with_addr",
// CHECK-SAME:                 flags: DIFlagStaticMember
// CHECK-SAME:                 extraData: i32 25

// CHECK:      ![[INT_DECL2:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_int2",
// CHECK-SAME:                         flags: DIFlagStaticMember
// CHECK-SAME:                         extraData: i32 26

// CHECK:      ![[FLOAT_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_float",
// CHECK-SAME:                          flags: DIFlagStaticMember
// CHECK-SAME:                          extraData: float

// CHECK:      ![[ENUM_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_enum",
// CHECK-SAME:                         flags: DIFlagStaticMember
// CHECK-SAME:                         extraData: i32 -1

// CHECK:      ![[EMPTY_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_struct_with_addr",
// CHECK-SAME:                          flags: DIFlagStaticMember
// CHECK-NOT:                           extraData:

// CHECK:      ![[EMPTY_TEMPLATED_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "empty_templated",
// CHECK-SAME:                                    flags: DIFlagStaticMember
// CHECK-SAME:                                    extraData: i32 1

// CHECK:      ![[TEMPLATE_DECL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_member, name: "cexpr_template",
// CHECK-SAME:                             flags: DIFlagStaticMember
// CHECK-SAME:                             extraData: i32 1

// CHECK:      !DIGlobalVariableExpression(var: ![[EMPTY_VAR:[0-9]+]], expr: !DIExpression())
// CHECK:      ![[EMPTY_VAR]] = distinct !DIGlobalVariable(name: "cexpr_struct_with_addr", linkageName:
// CHECK-SAME:                  isLocal: false, isDefinition: true, declaration: ![[EMPTY_DECL]])

// CHECK:      !DIGlobalVariableExpression(var: ![[INT_VAR2:[0-9]+]], expr: !DIExpression(DW_OP_constu, 26, DW_OP_stack_value))
// CHECK:      ![[INT_VAR2]] = distinct !DIGlobalVariable(name: "cexpr_int2"
// CHECK-NOT:                  linkageName:
// CHECK-SAME:                 isLocal: true, isDefinition: true, declaration: ![[INT_DECL2]])

// CHECK:      !DIGlobalVariableExpression(var: ![[FLOAT_VAR:[0-9]+]], expr: !DIExpression(DW_OP_constu, {{.*}}, DW_OP_stack_value))
// CHECK:      ![[FLOAT_VAR]] = distinct !DIGlobalVariable(name: "cexpr_float"
// CHECK-NOT:                   linkageName:
// CHECK-SAME:                  isLocal: true, isDefinition: true, declaration: ![[FLOAT_DECL]])

// CHECK:      !DIGlobalVariableExpression(var: ![[ENUM_VAR:[0-9]+]], expr: !DIExpression(DW_OP_constu, {{.*}}, DW_OP_stack_value))
// CHECK:      ![[ENUM_VAR]] = distinct !DIGlobalVariable(name: "cexpr_enum"
// CHECK-NOT:                  linkageName:
// CHECK-SAME:                 isLocal: true, isDefinition: true, declaration: ![[ENUM_DECL]])

// CHECK:      !DIGlobalVariableExpression(var: ![[EMPTY_TEMPLATED_VAR:[0-9]+]], expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
// CHECK:      ![[EMPTY_TEMPLATED_VAR]] = distinct !DIGlobalVariable(name: "empty_templated"
// CHECK-NOT:                             linkageName:
// CHECK-SAME:                            isLocal: true, isDefinition: true, declaration: ![[EMPTY_TEMPLATED_DECL]])

// CHECK:      !DIGlobalVariableExpression(var: ![[TEMPLATE_VAR:[0-9]+]], expr: !DIExpression(DW_OP_constu, 5, DW_OP_stack_value))
// CHECK:      ![[TEMPLATE_VAR]] = distinct !DIGlobalVariable(name: "cexpr_template"
// CHECK-NOT:                      linkageName:
// CHECK-SAME:                     isLocal: true, isDefinition: true, declaration: ![[TEMPLATE_DECL]], templateParams: ![[TEMPLATE_PARMS_2:[0-9]+]])
