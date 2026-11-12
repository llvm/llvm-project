// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

/** Documentation for NSUInteger */
typedef unsigned int NSUInteger;

/** Documentation for MyEnum */
typedef enum : NSUInteger {
        MyEnumFoo, /**< value Foo */
        MyEnumBar, /**< value Bar */
        MyEnumBaz, /**< value Baz */
} MyEnum;
// CHECK: TypedefDecl=MyEnum:[[@LINE-1]]:3 (Definition) {{.*}} FullCommentAsHTML=[<p class="para-brief"> Documentation for MyEnum </p>] FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-1]]" column="3"><Name>MyEnum</Name><USR>c:@T@MyEnum</USR><Declaration>typedef enum MyEnum MyEnum</Declaration><Abstract><Para> Documentation for MyEnum </Para></Abstract></Typedef>]


/** Documentation for E */
enum E {
        E_MyEnumFoo, /**< value Foo */
        E_MyEnumBar, /**< value Bar */
        E_MyEnumBaz, /**< value Baz */
};
typedef enum E E_T;
// CHECK: EnumDecl=E:[[@LINE-6]]:6 (Definition) {{.*}} BriefComment=[Documentation for E] FullCommentAsHTML=[<p class="para-brief"> Documentation for E </p>] FullCommentAsXML=[<Enum file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-6]]" column="6"><Name>E</Name><USR>c:@E@E</USR><Declaration>enum E{{( : int)?}} {}</Declaration><Abstract><Para> Documentation for E </Para></Abstract></Enum>]
// CHECK: TypedefDecl=E_T:[[@LINE-2]]:16 (Definition) FullCommentAsHTML=[<p class="para-brief"> Documentation for E </p>] FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-2]]" column="16"><Name>E</Name><USR>c:@E@E</USR><Declaration>typedef enum E E_T</Declaration><Abstract><Para> Documentation for E </Para></Abstract></Typedef>]


/** Comment about Foo */
typedef struct {
         int iii;
        } Foo;
// CHECK: TypedefDecl=Foo:[[@LINE-1]]:11 (Definition) {{.*}} FullCommentAsHTML=[<p class="para-brief"> Comment about Foo </p>] FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-1]]" column="11"><Name>Foo</Name><USR>c:@T@Foo</USR><Declaration>typedef struct Foo Foo</Declaration><Abstract><Para> Comment about Foo </Para></Abstract></Typedef>]
// CHECK: StructDecl=Foo:[[@LINE-4]]:9 (Definition) {{.*}} BriefComment=[Comment about Foo] FullCommentAsHTML=[<p class="para-brief"> Comment about Foo </p>] FullCommentAsXML=[<Class file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-4]]" column="9"><Name>&lt;anonymous&gt;</Name><USR>c:@SA@Foo</USR><Declaration>struct {}</Declaration><Abstract><Para> Comment about Foo </Para></Abstract></Class>]


struct Foo1 {
  int iii;
};
/** About Foo1T */
typedef struct Foo1 Foo1T;
// FIXME: we don't attach this comment to 'struct Foo1'
// CHECK: TypedefDecl=Foo1T:[[@LINE-2]]:21 (Definition) {{.*}} FullCommentAsHTML=[<p class="para-brief"> About Foo1T </p>] FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-2]]" column="21"><Name>Foo1T</Name><USR>c:annotate-comments-typedef.m@T@Foo1T</USR><Declaration>typedef struct Foo1 Foo1T</Declaration><Abstract><Para> About Foo1T </Para></Abstract></Typedef>]

