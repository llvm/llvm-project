// RUN: c-index-test -test-load-source all %s -fno-delayed-template-parsing | FileCheck %s
template<typename MyBase>
struct Derived:  MyBase::InnerIterator
{
    Derived() : MyBase::InnerIterator() {}
// CHECK:  TypeRef=MyBase:2:19 Extent=[5:17 - 5:23]
};

template<typename MyBase>
struct Derived2:  MyBase::Deeper::InnerIterator
{
    Derived2() : MyBase::Deeper::InnerIterator() {}
// CHECK:  TypeRef=MyBase:9:19 Extent=[12:18 - 12:24]
};

template<typename Q>
struct Templ;

template<typename MyBase>
struct Derived3:  Templ<MyBase>::InnerIterator
{
    Derived3() : Templ<MyBase>::InnerIterator() {}
// CHECK: TemplateRef=Templ:17:8 Extent=[22:18 - 22:23]
// CHECK: TypeRef=MyBase:19:19 Extent=[22:24 - 22:30]
};


struct Outer {
    template <typename Q>
    struct Inner {
        typedef Q Parm;
    };
};

template<typename Q>
struct Derived4:  Outer::Inner<Q>::Parm
{
    Derived4() : Outer::Inner<Q>::Parm() {}
// CHECK: TypeRef=struct Outer:28:8 Extent=[38:18 - 38:23]
// CHECK: TemplateRef=Inner:30:12 Extent=[38:25 - 38:30]
// CHECK: TypeRef=Q:35:19 Extent=[38:31 - 38:32]
};

template<typename Q>
struct Derived5:  Outer::Inner<Q>::Parm::InnerIterator
{
    Derived5() : Outer::Inner<Q>::Parm::InnerIterator() {}
// CHECK: TypeRef=struct Outer:28:8 Extent=[47:18 - 47:23]
// CHECK: TemplateRef=Inner:30:12 Extent=[47:25 - 47:30]
// CHECK: TypeRef=Q:44:19 Extent=[47:31 - 47:32]
};

template<typename Q>
struct Derived6:  Outer::Inner<Q>
{
    Derived6() : Outer::Inner<Q>() {}
// CHECK: TypeRef=struct Outer:28:8 Extent=[56:18 - 56:23]
// CHECK: TemplateRef=Inner:30:12 Extent=[56:25 - 56:30]
// CHECK: TypeRef=Q:53:19 Extent=[56:31 - 56:32]
};

struct Base {};

struct Derived7:  Outer::Inner<Base>::Parm
{
    Derived7() : Outer::Inner<Base>::Parm() {}
// CHECK: TypeRef=struct Outer:28:8 Extent=[66:18 - 66:23]
// CHECK: TemplateRef=Inner:30:12 Extent=[66:25 - 66:30]
// CHECK: TypeRef=struct Base:62:8 Extent=[66:31 - 66:35]
};

struct Derived8:  Outer::Inner<Base>
{
    Derived8() : Outer::Inner<Base>() {}
// CHECK: TypeRef=struct Outer:28:8 Extent=[74:18 - 74:23]
// CHECK: TemplateRef=Inner:30:12 Extent=[74:25 - 74:30]
// CHECK: TypeRef=struct Base:62:8 Extent=[74:31 - 74:35]
};

namespace Namespace {
    template<typename Q> struct Templ;

    struct Outer {
        template <typename Q>
        struct Inner {
            typedef Q Parm;
        };
    };
}

template<typename MyBase>
struct Derived9:  Namespace::Templ<MyBase>::InnerIterator
{
    Derived9() : Namespace::Templ<MyBase>::InnerIterator() {}
// CHECK: NamespaceRef=Namespace:80:11 Extent=[94:18 - 94:27]
// CHECK: TemplateRef=Templ:81:33 Extent=[94:29 - 94:34]
// CHECK: TypeRef=MyBase:91:19 Extent=[94:35 - 94:41]
};

template<typename MyBase>
struct Derived10:  Namespace::Templ<MyBase>
{
    Derived10() : Namespace::Templ<MyBase>() {}
// CHECK: NamespaceRef=Namespace:80:11 Extent=[103:19 - 103:28]
// CHECK: TemplateRef=Templ:81:33 Extent=[103:30 - 103:35]
// CHECK: TypeRef=MyBase:100:19 Extent=[103:36 - 103:42]
};

template<typename MyBase>
struct Derived11:  Namespace::Outer::Inner<MyBase>::Parm
{
    Derived11() : Namespace::Outer::Inner<MyBase>::Parm() {}
// CHECK: NamespaceRef=Namespace:80:11 Extent=[112:19 - 112:28]
// CHECK: TypeRef=struct Namespace::Outer:83:12 Extent=[112:30 - 112:35]
// CHECK: TemplateRef=Inner:85:16 Extent=[112:37 - 112:42]
// CHECK: TypeRef=MyBase:109:19 Extent=[112:43 - 112:49]
};
